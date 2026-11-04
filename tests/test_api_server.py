"""Tests for the FastAPI server."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import soundfile as sf
import torch
from fastapi.testclient import TestClient

from omnivoice_triton.cli.api_server import build_parser, create_app
from omnivoice_triton.serving import QueueFullError, WorkerLifecycleReporter


class _FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []

    def __call__(
        self,
        text: str,
        *,
        return_tensors: str = "pt",
        add_special_tokens: bool = True,
    ) -> SimpleNamespace:
        del return_tensors
        self.calls.append((text, add_special_tokens))
        token_count = max(1, len(text.split()))
        return SimpleNamespace(input_ids=torch.ones((1, token_count), dtype=torch.long))


def _fake_wav_bytes() -> bytes:
    buffer = BytesIO()
    sf.write(
        buffer,
        np.zeros(1600, dtype=np.float32),
        16000,
        format="WAV",
        subtype="PCM_16",
    )
    return buffer.getvalue()


class _FakeGraphForward:
    """Stands in for the CUDA Graph forward wrapper: records which thread clears it."""

    def __init__(self) -> None:
        self.clear_threads: list[str] = []

    def clear(self) -> None:
        self.clear_threads.append(threading.current_thread().name)


class _FakeModel:
    def __init__(self, *, generate_delay_s: float = 0.0) -> None:
        self.sampling_rate = 24000
        self._asr_pipe = None
        self.config = SimpleNamespace(num_audio_codebook=8, audio_mask_id=1024)
        self.text_tokenizer = _FakeTokenizer()
        self.audio_tokenizer = SimpleNamespace(
            config=SimpleNamespace(frame_rate=75),
        )
        self.calls: list[dict[str, Any]] = []
        self.prompt_calls: list[dict[str, Any]] = []
        self.prompt_threads: list[str] = []
        self.prompt_grad_modes: list[tuple[bool, bool]] = []
        self.prompt_error: BaseException | None = None
        self.generate_error: BaseException | None = None
        self.forward = _FakeGraphForward()
        self.generate_delay_s = generate_delay_s
        self._last_generation_metrics = {
            "generate_total_ms": 11.0,
            "prepare_inference_inputs_ms": 1.5,
            "iterative_generate_ms": 7.0,
            "chunked_generate_ms": 0.0,
            "decode_postprocess_ms": 2.0,
        }

    def load_asr_model(self) -> None:
        self._asr_pipe = object()

    def _estimate_target_tokens(
        self,
        text: str,
        ref_text: str | None,
        num_ref_audio_tokens: int | None,
        *,
        speed: float = 1.0,
    ) -> int:
        del ref_text
        base = max(1, len(text.split()) * 12)
        if num_ref_audio_tokens is not None:
            base += min(num_ref_audio_tokens // 4, 20)
        return max(1, int(base / speed))

    def create_voice_clone_prompt(
        self,
        *,
        ref_audio: Any,
        ref_text: str | None,
        preprocess_prompt: bool,
    ) -> SimpleNamespace:
        if isinstance(ref_audio, tuple):
            waveform, sample_rate = ref_audio
            ref_audio_meta = {
                "shape": tuple(waveform.shape),
                "sample_rate": sample_rate,
            }
        else:
            ref_audio_meta = {"value": ref_audio}
        call = {
            "ref_audio": ref_audio_meta,
            "ref_text": ref_text,
            "preprocess_prompt": preprocess_prompt,
        }
        self.prompt_calls.append(call)
        self.prompt_threads.append(threading.current_thread().name)
        self.prompt_grad_modes.append(
            (torch.is_grad_enabled(), torch.is_inference_mode_enabled())
        )
        if self.prompt_error is not None:
            raise self.prompt_error
        return SimpleNamespace(
            ref_audio_tokens=torch.zeros((8, 48), dtype=torch.long),
            ref_text=ref_text or "Auto transcript.",
            ref_rms=0.2,
        )

    def generate(self, **kwargs: Any) -> list[np.ndarray]:
        self.calls.append(kwargs)
        if self.generate_error is not None:
            raise self.generate_error
        texts = kwargs["text"]
        if isinstance(texts, str):
            texts = [texts]
        if self.generate_delay_s > 0:
            time.sleep(self.generate_delay_s)
        return [
            np.full(2400, 0.1 + i * 0.05, dtype=np.float32) for i, _ in enumerate(texts)
        ]

    def get_runtime_stage_metrics(self) -> dict[str, float]:
        return dict(self._last_generation_metrics)


class _FakeRunner:
    def __init__(
        self,
        name: str,
        *,
        model_generate_delay_s: float = 0.0,
        **kwargs: Any,
    ) -> None:
        self.name = name
        self.kwargs = kwargs
        self.loaded = False
        self.unloaded = False
        self._model = _FakeModel(generate_delay_s=model_generate_delay_s)
        self.prewarmed_shapes: list[list[int]] = []

    def load_model(self) -> None:
        self.loaded = True

    @property
    def model(self) -> _FakeModel:
        return self._model

    def unload_model(self) -> None:
        self.unloaded = True

    def get_cuda_graph_metrics(self) -> dict[str, Any]:
        return {"capture_count": 0, "shapes": []}

    def get_generation_metrics(self) -> dict[str, float]:
        return self._model.get_runtime_stage_metrics()

    def prewarm_cuda_graph_shapes(
        self,
        shapes: list[tuple[int, int, int]] | tuple[tuple[int, int, int], ...],
    ) -> dict[str, Any]:
        self.prewarmed_shapes = [list(shape) for shape in shapes]
        return {
            "capture_count": len(self.prewarmed_shapes),
            "shapes": list(self.prewarmed_shapes),
            "warmed_shapes": list(self.prewarmed_shapes),
        }


def _make_app(
    tmp_path: Path,
    created: list[_FakeRunner],
    *,
    model_generate_delay_s: float = 0.0,
    **kwargs: Any,
):
    def _factory(name: str, **factory_kwargs: Any) -> _FakeRunner:
        runner = _FakeRunner(
            name,
            model_generate_delay_s=model_generate_delay_s,
            **factory_kwargs,
        )
        created.append(runner)
        return runner

    return create_app(
        model_checkpoint="fake/omnivoice",
        runner_name="hybrid",
        device="cuda",
        dtype="fp16",
        runner_factory=_factory,
        save_dir=str(tmp_path),
        **kwargs,
    )


def test_health_and_languages(tmp_path: Path) -> None:
    created: list[_FakeRunner] = []
    app = _make_app(
        tmp_path,
        created,
        load_asr=True,
        enable_sage_attention=True,
        full_triton_patch=True,
        decode_postprocess_workers=3,
        audio_decode_batch_size=8,
        prewarm_clone_batch_sizes=(2, 4),
        prewarm_clone_sequence_lengths=(163,),
    )

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        payload = health.json()
        assert payload["status"] == "ok"
        assert payload["runner"] == "hybrid"
        assert payload["model"] == "fake/omnivoice"
        assert payload["device"] == "cuda"
        assert payload["allow_lazy_asr"] is False
        assert payload["asr_loaded"] is True
        assert payload["sage_attention"] is True
        assert payload["full_triton_patch"] is True
        assert payload["decode_postprocess_workers"] == 3
        assert payload["audio_decode_batch_size"] == 8
        assert payload["max_cuda_graphs"] == 8
        assert payload["batch_collect_ms"] == 10.0
        assert payload["max_batch_requests"] == 32
        assert payload["registered_clone_prompt_count"] == 0
        assert payload["token_estimate_cache_entries"] == 0
        assert payload["batch_bucket_sizes"] == [1, 2, 4, 8, 16, 32]
        assert payload["cuda_graph_capture_count"] == 0
        assert payload["cuda_graph_max_graphs"] == 8
        assert payload["prewarm_clone_batch_sizes"] == [2, 4]
        assert payload["prewarm_clone_sequence_lengths"] == [163]
        assert payload["cuda_graph_prewarm"]["requested_shapes"] == [
            [4, 8, 163],
            [8, 8, 163],
        ]
        assert payload["cuda_graph_prewarm"]["warmed_shapes"] == [
            [4, 8, 163],
            [8, 8, 163],
        ]
        assert payload["last_generation_metrics"]["generate_total_ms"] == 11.0
        assert {"available"} <= set(payload["gpu_metrics"])
        assert payload["last_batch_gpu_metrics"] is None
        assert payload["worker"]["worker_type"] == "tts"
        assert payload["worker"]["status"] == "healthy"
        assert payload["worker"]["active_requests"] == 0
        assert payload["worker"]["assignable"] is True
        assert payload["worker"]["callback_enabled"] is False
        assert Path(payload["save_dir"]) == tmp_path

        healthz = client.get("/healthz")
        assert healthz.status_code == 200
        assert healthz.json() == {
            "ok": True,
            "service": "omnivoice",
            "worker_type": "tts",
            "status": "healthy",
        }

        languages = client.get("/languages")
        assert languages.status_code == 200
        lang_payload = languages.json()
        assert lang_payload["count"] > 0
        assert {"id", "name", "display_name"} <= set(lang_payload["languages"][0])

    assert created[0].loaded is True
    assert created[0].unloaded is True
    assert created[0].kwargs["enable_sage_attention"] is True
    assert created[0].kwargs["patch_range"] is None
    assert created[0].kwargs["decode_postprocess_workers"] == 3
    assert created[0].kwargs["audio_decode_batch_size"] == 8
    assert created[0].kwargs["max_cuda_graphs"] == 8
    assert created[0].prewarmed_shapes == [[4, 8, 163], [8, 8, 163]]


def test_audio_decode_batch_size_defaults_to_serial_and_validates() -> None:
    parser = build_parser()
    assert parser.parse_args([]).audio_decode_batch_size == 1
    assert (
        parser.parse_args(["--audio-decode-batch-size", "8"]).audio_decode_batch_size
        == 8
    )
    with pytest.raises(ValueError, match="audio_decode_batch_size"):
        create_app(audio_decode_batch_size=0)


def test_worker_drain_rejects_new_generate_requests(tmp_path: Path) -> None:
    created: list[_FakeRunner] = []
    app = _make_app(
        tmp_path,
        created,
        load_asr=False,
        start_worker_callback=False,
    )

    with TestClient(app) as client:
        drain = client.post("/drain")
        assert drain.status_code == 200
        drain_payload = drain.json()
        assert drain_payload["status"] == "draining"
        assert drain_payload["assignable"] is False

        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["worker"]["status"] == "draining"

        response = client.post(
            "/generate",
            data={"mode": "design", "text": "No new work.", "instruct": "warm"},
        )
        assert response.status_code == 503
        assert response.json() == {"ok": False, "error": "worker_draining"}


def test_worker_drain_rejects_new_prompt_and_cache_work(tmp_path: Path) -> None:
    created: list[_FakeRunner] = []
    app = _make_app(
        tmp_path,
        created,
        load_asr=False,
        start_worker_callback=False,
    )

    with TestClient(app) as client:
        drain = client.post("/drain")
        assert drain.status_code == 200

        clone_prompt = client.post(
            "/clone-prompts",
            files={"ref_audio": ("ref.wav", _fake_wav_bytes(), "audio/wav")},
            data={"ref_text": "Reference."},
        )
        assert clone_prompt.status_code == 503
        assert clone_prompt.json() == {"ok": False, "error": "worker_draining"}

        warm = client.post(
            "/cache/warm?wait=true",
            files={"ref_audio": ("ref.wav", _fake_wav_bytes(), "audio/wav")},
            data={"mode": "clone", "ref_text": "Reference."},
        )
        assert warm.status_code == 503
        assert warm.json() == {"ok": False, "error": "worker_draining"}


def test_worker_drain_requires_token_when_configured(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("LINGUA_CONTROL_PLANE_BASE_URL", "https://lingua.example")
    monkeypatch.setenv("LINGUA_WORKER_TOKEN", "secret-token")
    monkeypatch.setenv("LINGUA_WORKER_PUBLIC_BASE_URL", "http://203.0.113.10:8002")
    monkeypatch.setenv("VAST_INSTANCE_ID", "12345")

    created: list[_FakeRunner] = []
    app = _make_app(
        tmp_path,
        created,
        load_asr=False,
        start_worker_callback=False,
    )

    with TestClient(app) as client:
        unauthorized = client.post("/drain")
        assert unauthorized.status_code == 401

        authorized = client.post(
            "/drain",
            headers={"X-Worker-Token": "secret-token"},
        )
        assert authorized.status_code == 200
        payload = authorized.json()
        assert payload["worker_type"] == "tts"
        assert payload["worker_id"] == "tts-12345"
        assert payload["base_url"] == "http://203.0.113.10:8002"
        assert payload["callback_enabled"] is True


def test_worker_state_reads_vast_public_port_for_internal_port(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("LINGUA_CONTROL_PLANE_BASE_URL", "https://lingua.example")
    monkeypatch.setenv("LINGUA_WORKER_TOKEN", "secret-token")
    monkeypatch.setenv("PUBLIC_IPADDR", "74.48.140.178")
    monkeypatch.setenv("VAST_TCP_PORT_8000", "43538")

    created: list[_FakeRunner] = []
    app = _make_app(
        tmp_path,
        created,
        load_asr=False,
        server_port=8000,
        start_worker_callback=False,
    )

    with TestClient(app) as client:
        response = client.get("/worker/state")
        assert response.status_code == 200
        payload = response.json()
        assert payload["base_url"] == "http://74.48.140.178:43538"
        assert payload["endpoint_url"] == "http://74.48.140.178:43538"
        assert payload["public_ip"] == "74.48.140.178"
        assert payload["public_port"] == "43538"
        assert payload["internal_port"] == 8000

        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["worker"]["base_url"] == payload["base_url"]


def test_worker_callback_payload_matches_lingua_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("LINGUA_CONTROL_PLANE_BASE_URL", "https://lingua.example")
    monkeypatch.setenv("LINGUA_WORKER_TOKEN", "secret-token")
    monkeypatch.setenv("PUBLIC_IPADDR", "74.48.140.178")
    monkeypatch.setenv("VAST_TCP_PORT_8000", "43538")
    monkeypatch.setenv("VAST_INSTANCE_ID", "12345")
    monkeypatch.setenv("OMNIVOICE_MODEL_ID", "omnivoice-test")
    monkeypatch.setenv("OMNIVOICE_MAX_CONCURRENT", "1")

    created: list[_FakeRunner] = []
    app = _make_app(
        tmp_path,
        created,
        load_asr=False,
        server_port=8000,
        start_worker_callback=False,
    )

    with TestClient(app):
        reporter = WorkerLifecycleReporter(
            app.state.worker_callback_config,
            app.state.worker_runtime,
        )
        payload = reporter.payload()

    assert payload["worker_type"] == "tts"
    assert payload["worker_id"] == "tts-12345"
    assert payload["instance_id"] == "12345"
    assert payload["base_url"] == "http://74.48.140.178:43538"
    assert payload["capacity"] == 1
    assert payload["status"] == "healthy"
    assert payload["metadata"] == {
        "provider": "omnivoice",
        "model_id": "omnivoice-test",
        "queue_depth": 0,
        "active_requests": 0,
        "max_concurrent": 1,
        "gpu_utilization": 0,
        "gpu_memory_pct": 0,
        "p95_latency_ms": 0,
    }


def test_worker_base_url_override_wins(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("LINGUA_CONTROL_PLANE_BASE_URL", "https://lingua.example")
    monkeypatch.setenv("LINGUA_WORKER_TOKEN", "secret-token")
    monkeypatch.setenv("LINGUA_WORKER_BASE_URL", "http://203.0.113.55:45678")
    monkeypatch.setenv("PUBLIC_IPADDR", "74.48.140.178")
    monkeypatch.setenv("VAST_TCP_PORT_8000", "43538")

    created: list[_FakeRunner] = []
    app = _make_app(
        tmp_path,
        created,
        load_asr=False,
        server_port=8000,
        start_worker_callback=False,
    )

    with TestClient(app) as client:
        response = client.get("/worker/state")
        assert response.status_code == 200
        payload = response.json()
        assert payload["base_url"] == "http://203.0.113.55:45678"
        assert payload["endpoint_url"] == "http://203.0.113.55:45678"
        assert payload["internal_port"] == 8000


def test_worker_state_does_not_advertise_unmapped_vast_internal_port(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("LINGUA_CONTROL_PLANE_BASE_URL", "https://lingua.example")
    monkeypatch.setenv("LINGUA_WORKER_TOKEN", "secret-token")
    monkeypatch.setenv("PUBLIC_IPADDR", "74.48.140.178")
    monkeypatch.setenv("VAST_TCP_PORT_8000", "43538")

    created: list[_FakeRunner] = []
    app = _make_app(
        tmp_path,
        created,
        load_asr=False,
        server_port=8002,
        start_worker_callback=False,
    )

    with TestClient(app) as client:
        response = client.get("/worker/state")
        assert response.status_code == 200
        payload = response.json()
        assert payload["base_url"] is None
        assert payload["endpoint_url"] is None
        assert payload["public_ip"] == "74.48.140.178"
        assert payload["public_port"] is None
        assert payload["internal_port"] == 8002


def test_generate_design_returns_wav_and_saves_file(tmp_path: Path) -> None:
    created: list[_FakeRunner] = []
    app = _make_app(tmp_path, created, load_asr=False)

    with TestClient(app) as client:
        response = client.post(
            "/generate",
            data={
                "mode": "design",
                "text": "Hello from the API.",
                "instruct": "female, young adult",
                "num_step": "16",
                "guidance_scale": "1.5",
                "position_temperature": "0.0",
                "class_temperature": "0.0",
                "duration": "1.2",
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    assert response.headers["x-omnivoice-latency-ms"]
    assert response.headers["x-omnivoice-pre-batch-ms"]
    assert response.headers["x-omnivoice-batch-estimate-ms"]
    assert response.headers["x-omnivoice-response-encode-ms"]
    assert response.headers["x-omnivoice-peak-vram-gb"]
    assert response.headers["x-omnivoice-audio-duration-s"]
    assert response.headers["x-omnivoice-batch-requests"] == "1"
    assert response.headers["x-omnivoice-prompt-source"] == "none"
    assert response.headers["x-omnivoice-postprocess-mode"] == "full"
    assert response.headers["x-omnivoice-generate-total-ms"] == "11.00"
    assert response.headers["x-omnivoice-prepare-inference-inputs-ms"] == "1.50"
    assert response.headers["x-omnivoice-iterative-generate-ms"] == "7.00"
    assert response.headers["x-omnivoice-decode-postprocess-ms"] == "2.00"
    if "x-omnivoice-gpu-sample-count" in response.headers:
        assert int(response.headers["x-omnivoice-gpu-sample-count"]) >= 0
    saved_path = Path(response.headers["x-omnivoice-saved-path"])
    assert saved_path.exists()

    model = created[0].model
    call = model.calls[0]
    assert call["text"] == ["Hello from the API."]
    assert call["instruct"] == ["female, young adult"]
    assert call["duration"] == [1.2]
    assert call["generation_config"].num_step == 16
    assert call["generation_config"].guidance_scale == 1.5


def test_generate_design_without_instruct_matches_gradio_fallback(
    tmp_path: Path,
) -> None:
    created: list[_FakeRunner] = []
    app = _make_app(tmp_path, created, load_asr=False)

    with TestClient(app) as client:
        response = client.post(
            "/generate",
            data={
                "mode": "design",
                "text": "Fallback to auto-style generation.",
                "duration": "0",
            },
        )

    assert response.status_code == 200
    call = created[0].model.calls[0]
    assert "instruct" not in call
    assert "duration" not in call


def test_generate_clone_allows_instruct_and_prompt_options(tmp_path: Path) -> None:
    created: list[_FakeRunner] = []
    app = _make_app(tmp_path, created, load_asr=False)

    with TestClient(app) as client:
        response = client.post(
            "/generate",
            data={
                "mode": "clone",
                "text": "Clone this voice.",
                "ref_text": "Reference transcript.",
                "instruct": "whisper",
                "preprocess_prompt": "false",
            },
            files={"ref_audio": ("ref.wav", _fake_wav_bytes(), "audio/wav")},
        )

    assert response.status_code == 200
    model = created[0].model
    prompt_call = model.prompt_calls[0]
    assert prompt_call["ref_text"] == "Reference transcript."
    assert prompt_call["preprocess_prompt"] is False

    generate_call = model.calls[0]
    assert generate_call["instruct"] == ["whisper"]
    assert "voice_clone_prompt" in generate_call
    assert len(generate_call["voice_clone_prompt"]) == 1


def test_generate_clone_reuses_cached_prompt(tmp_path: Path) -> None:
    created: list[_FakeRunner] = []
    app = _make_app(tmp_path, created, load_asr=False)

    request_kwargs = {
        "data": {
            "mode": "clone",
            "text": "Clone this voice again.",
            "ref_text": "Reference transcript.",
        },
        "files": {"ref_audio": ("ref.wav", _fake_wav_bytes(), "audio/wav")},
    }

    with TestClient(app) as client:
        first = client.post("/generate", **request_kwargs)
        second = client.post("/generate", **request_kwargs)

    assert first.status_code == 200
    assert second.status_code == 200
    model = created[0].model
    assert len(model.prompt_calls) == 1
    assert len(model.calls) == 2
    assert first.headers["x-omnivoice-prompt-source"] == "upload_miss"
    assert second.headers["x-omnivoice-prompt-source"] == "upload_hit"


def test_cache_warm_wait_true_populates_clone_prompt_cache(tmp_path: Path) -> None:
    created: list[_FakeRunner] = []
    app = _make_app(tmp_path, created, load_asr=False)
    ref_audio_bytes = _fake_wav_bytes()

    with TestClient(app) as client:
        warm = client.post(
            "/cache/warm?wait=true",
            data={
                "mode": "clone",
                "language": "en",
                "ref_text": "Reference transcript.",
                "num_step": "16",
                "denoise": "false",
                "preprocess_prompt": "false",
                "postprocess_mode": "light",
            },
            files={"ref_audio": ("ref.wav", ref_audio_bytes, "audio/wav")},
        )
        assert warm.status_code == 200
        payload = warm.json()
        assert payload["status"] == "ok"
        assert payload["cache"] == "clone_prompt"
        assert payload["source"] == "miss"
        assert payload["warmed"] is True
        assert payload["wait"] is True
        assert payload["language"] == "en"
        assert payload["num_step"] == 16
        assert payload["denoise"] is False
        assert payload["postprocess_mode"] == "light"

        response = client.post(
            "/generate",
            data={
                "mode": "clone",
                "text": "Use the warmed prompt.",
                "ref_text": "Reference transcript.",
                "preprocess_prompt": "false",
            },
            files={"ref_audio": ("ref.wav", ref_audio_bytes, "audio/wav")},
        )

    assert response.status_code == 200
    assert response.headers["x-omnivoice-prompt-source"] == "upload_hit"
    assert len(created[0].model.prompt_calls) == 1


def test_cache_warm_wait_false_runs_in_background(tmp_path: Path) -> None:
    created: list[_FakeRunner] = []
    app = _make_app(tmp_path, created, load_asr=False)
    ref_audio_bytes = _fake_wav_bytes()

    with TestClient(app) as client:
        model = created[0].model
        original_create_prompt = model.create_voice_clone_prompt
        started = threading.Event()
        release = threading.Event()

        def _slow_create_prompt(**kwargs: Any) -> SimpleNamespace:
            started.set()
            release.wait(timeout=2.0)
            return original_create_prompt(**kwargs)

        model.create_voice_clone_prompt = _slow_create_prompt
        try:
            warm = client.post(
                "/cache/warm?wait=false",
                data={
                    "mode": "clone",
                    "ref_text": "Reference transcript.",
                },
                files={"ref_audio": ("ref.wav", ref_audio_bytes, "audio/wav")},
            )
            assert warm.status_code == 200
            assert warm.json()["status"] == "accepted"
            assert started.wait(timeout=1.0)

            health = client.get("/health")
            assert health.status_code == 200
            assert health.json()["worker"]["active_requests"] == 1
        finally:
            release.set()

        deadline = time.time() + 2.0
        while len(model.prompt_calls) < 1 and time.time() < deadline:
            time.sleep(0.01)

        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["worker"]["active_requests"] == 0
        assert len(model.prompt_calls) == 1


def test_cache_warm_requires_ref_text_when_lazy_asr_disabled(
    tmp_path: Path,
) -> None:
    created: list[_FakeRunner] = []
    app = _make_app(tmp_path, created, load_asr=False)

    with TestClient(app) as client:
        response = client.post(
            "/cache/warm?wait=true",
            data={"mode": "clone"},
            files={"ref_audio": ("ref.wav", _fake_wav_bytes(), "audio/wav")},
        )

    assert response.status_code == 400
    assert "ref_text is required" in response.json()["detail"]
    assert created[0].model.prompt_calls == []


def test_generate_clone_without_ref_text_rejected_when_lazy_asr_disabled(
    tmp_path: Path,
) -> None:
    created: list[_FakeRunner] = []
    app = _make_app(tmp_path, created, load_asr=False)

    with TestClient(app) as client:
        response = client.post(
            "/generate",
            data={
                "mode": "clone",
                "text": "Clone without transcript.",
            },
            files={"ref_audio": ("ref.wav", _fake_wav_bytes(), "audio/wav")},
        )

    assert response.status_code == 400
    assert "ref_text is required" in response.json()["detail"]
    assert created[0].model._asr_pipe is None
    assert created[0].model.prompt_calls == []


def test_generate_design_accepts_light_postprocess_mode(tmp_path: Path) -> None:
    created: list[_FakeRunner] = []
    app = _make_app(tmp_path, created, load_asr=False)

    with TestClient(app) as client:
        response = client.post(
            "/generate",
            data={
                "mode": "design",
                "text": "Use the light postprocess mode.",
                "instruct": "warm",
                "postprocess_output": "light",
            },
        )

    assert response.status_code == 200
    assert response.headers["x-omnivoice-postprocess-mode"] == "light"
    gen_config = created[0].model.calls[0]["generation_config"]
    assert gen_config.postprocess_output is False
    assert getattr(gen_config, "postprocess_mode") == "light"


def test_register_clone_prompt_and_generate_with_prompt_id(tmp_path: Path) -> None:
    created: list[_FakeRunner] = []
    app = _make_app(tmp_path, created, load_asr=False)

    with TestClient(app) as client:
        register = client.post(
            "/clone-prompts",
            data={"ref_text": "Reference transcript.", "preprocess_prompt": "false"},
            files={"ref_audio": ("ref.wav", _fake_wav_bytes(), "audio/wav")},
        )
        assert register.status_code == 200
        payload = register.json()
        prompt_id = payload["prompt_id"]

        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["registered_clone_prompt_count"] == 1
        cache_status = client.get(f"/cache/status?cache_key={prompt_id}")
        assert cache_status.status_code == 200
        assert cache_status.json()["status"] == "ready"
        assert cache_status.json()["cached"] is True
        assert cache_status.json()["prompt_id"] == prompt_id

        unknown_cache_status = client.get("/cache/status?cache_key=missing")
        assert unknown_cache_status.status_code == 200
        assert unknown_cache_status.json()["status"] == "unknown"
        assert unknown_cache_status.json()["cached"] is False

        response = client.post(
            "/generate",
            data={
                "mode": "clone",
                "text": "Use the registered prompt.",
                "prompt_id": prompt_id,
                "instruct": "steady",
            },
        )

    assert response.status_code == 200
    assert response.headers["x-omnivoice-prompt-source"] == "prompt_id"
    assert response.headers["x-omnivoice-prompt-id"] == prompt_id
    model = created[0].model
    assert len(model.prompt_calls) == 1
    assert len(model.calls) == 1
    assert model.calls[0]["instruct"] == ["steady"]


def test_token_estimate_cache_reuses_tokenizer_results(tmp_path: Path) -> None:
    created: list[_FakeRunner] = []
    app = _make_app(tmp_path, created, load_asr=False, token_estimate_cache_size=16)

    with TestClient(app) as client:
        for _ in range(2):
            response = client.post(
                "/generate",
                data={
                    "mode": "design",
                    "text": "Cache the tokenizer work.",
                    "instruct": "warm",
                },
            )
            assert response.status_code == 200

        health = client.get("/health")

    tokenizer = created[0].model.text_tokenizer
    assert len(tokenizer.calls) == 2
    assert health.status_code == 200
    assert health.json()["token_estimate_cache_entries"] == 2


def test_generate_batches_concurrent_requests(tmp_path: Path) -> None:
    created: list[_FakeRunner] = []
    app = _make_app(
        tmp_path,
        created,
        load_asr=False,
        batch_collect_ms=50.0,
        max_batch_requests=8,
        max_batch_target_tokens=10000,
        max_batch_conditioning_tokens=10000,
        max_batch_padding_ratio=10.0,
        model_generate_delay_s=0.05,
    )

    texts = [
        "Batch request number one.",
        "Batch request number two.",
        "Batch request number three.",
        "Batch request number four.",
    ]

    with TestClient(app) as client:
        start_barrier = threading.Barrier(len(texts))

        def _send(text: str):
            start_barrier.wait()
            return client.post(
                "/generate",
                data={"mode": "design", "text": text, "instruct": "warm"},
            )

        with ThreadPoolExecutor(max_workers=len(texts)) as executor:
            futures = [executor.submit(_send, text) for text in texts]
            responses = [future.result() for future in futures]

    assert all(response.status_code == 200 for response in responses)
    batch_sizes = [
        int(response.headers["x-omnivoice-batch-requests"]) for response in responses
    ]
    assert max(batch_sizes) >= 2

    model = created[0].model
    assert len(model.calls) < len(texts)
    flattened = [text for call in model.calls for text in call["text"]]
    assert sorted(flattened) == sorted(texts)


def test_waiting_generations_do_not_starve_health_or_hide_queue(
    tmp_path: Path,
) -> None:
    """More than AnyIO's default 40 waiters remain visible and health stays usable."""
    created: list[_FakeRunner] = []
    app = _make_app(
        tmp_path,
        created,
        load_asr=False,
        batch_collect_ms=0,
        max_pending_requests=64,
        max_batch_requests=1,
    )
    entered, release = threading.Event(), threading.Event()
    request_count = 48

    with TestClient(app) as client:
        original_generate = created[0].model.generate

        def blocked_generate(**kwargs):
            entered.set()
            assert release.wait(10)
            return original_generate(**kwargs)

        created[0].model.generate = blocked_generate
        with ThreadPoolExecutor(max_workers=request_count + 1) as executor:
            responses = [
                executor.submit(client.post, "/generate", **_design_request())
                for _ in range(request_count)
            ]
            try:
                assert entered.wait(3)
                deadline = time.monotonic() + 5
                while (
                    app.state.batcher.queue_snapshot()["pending_requests"]
                    < request_count - 1
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.01)
                assert (
                    app.state.batcher.queue_snapshot()["pending_requests"]
                    == request_count - 1
                )
                health = executor.submit(client.get, "/health").result(timeout=2)
                assert health.status_code == 200
                assert health.json()["worker"]["active_requests"] == request_count
                assert health.json()["batcher_queue"]["pending_requests"] == 47
            finally:
                release.set()
            assert all(f.result(timeout=5).status_code == 200 for f in responses)


def test_cold_prompt_waits_use_separate_bounded_cpu_limiter(tmp_path: Path) -> None:
    """Waiting for prompt creation must leave the control-route pool available."""
    created: list[_FakeRunner] = []
    app = _make_app(tmp_path, created, load_asr=False)
    entered, release = threading.Event(), threading.Event()

    def blocker():
        entered.set()
        assert release.wait(10)

    def upload(client):
        return client.post(
            "/generate",
            data={"mode": "clone", "text": "Hello.", "ref_text": "Reference."},
            files={"ref_audio": ("ref.wav", _fake_wav_bytes(), "audio/wav")},
        )

    with TestClient(app) as client:
        count = int(app.state.prompt_prepare_limiter.total_tokens) + 2
        with ThreadPoolExecutor(max_workers=count + 2) as executor:
            blocked = executor.submit(
                app.state.batcher.run_on_gpu_thread, blocker, name="blocker"
            )
            assert entered.wait(3)
            responses = [executor.submit(upload, client) for _ in range(count)]
            try:
                deadline = time.monotonic() + 5
                while (
                    app.state.prompt_prepare_limiter.borrowed_tokens < count - 2
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.01)
                assert app.state.prompt_prepare_limiter.borrowed_tokens == count - 2
                health = executor.submit(client.get, "/health").result(timeout=2)
                assert health.status_code == 200
                assert health.json()["batcher_queue"]["pending_gpu_tasks"] == 1
            finally:
                release.set()
            blocked.result(timeout=3)
            assert all(f.result(timeout=5).status_code == 200 for f in responses)


def test_generate_uses_batch_buckets(tmp_path: Path) -> None:
    created: list[_FakeRunner] = []
    app = _make_app(
        tmp_path,
        created,
        load_asr=False,
        batch_collect_ms=50.0,
        max_batch_requests=8,
        max_batch_target_tokens=10000,
        max_batch_conditioning_tokens=10000,
        max_batch_padding_ratio=10.0,
        batch_bucket_sizes=(1, 2, 4),
        model_generate_delay_s=0.05,
    )

    texts = [
        "Bucket request one.",
        "Bucket request two.",
        "Bucket request three.",
        "Bucket request four.",
        "Bucket request five.",
        "Bucket request six.",
    ]

    with TestClient(app) as client:
        start_barrier = threading.Barrier(len(texts))

        def _send(text: str):
            start_barrier.wait()
            return client.post(
                "/generate",
                data={"mode": "design", "text": text, "instruct": "warm"},
            )

        with ThreadPoolExecutor(max_workers=len(texts)) as executor:
            futures = [executor.submit(_send, text) for text in texts]
            responses = [future.result() for future in futures]

    assert all(response.status_code == 200 for response in responses)
    model = created[0].model
    batch_sizes = sorted(len(call["text"]) for call in model.calls)
    assert batch_sizes == [2, 4]


def test_generate_validation_errors(tmp_path: Path) -> None:
    created: list[_FakeRunner] = []
    app = _make_app(tmp_path, created, load_asr=False)

    with TestClient(app) as client:
        auto_with_instruct = client.post(
            "/generate",
            data={"mode": "auto", "text": "hello", "instruct": "female"},
        )
        assert auto_with_instruct.status_code == 400

        design_missing_instruct = client.post(
            "/generate",
            data={"mode": "design", "text": "hello"},
        )
        assert design_missing_instruct.status_code == 200

        clone_missing_audio = client.post(
            "/generate",
            data={"mode": "clone", "text": "hello"},
        )
        assert clone_missing_audio.status_code == 400

        clone_both_prompt_and_audio = client.post(
            "/generate",
            data={"mode": "clone", "text": "hello", "prompt_id": "abc123"},
            files={"ref_audio": ("ref.wav", _fake_wav_bytes(), "audio/wav")},
        )
        assert clone_both_prompt_and_audio.status_code == 400

        clone_prompt_with_ref_text = client.post(
            "/generate",
            data={
                "mode": "clone",
                "text": "hello",
                "prompt_id": "abc123",
                "ref_text": "nope",
            },
        )
        assert clone_prompt_with_ref_text.status_code == 400


@pytest.mark.parametrize("postprocess_mode", ["full", "light", "off"])
@pytest.mark.parametrize("mode", ["auto", "design", "clone"])
def test_insert_pauses_wav_and_options(
    tmp_path: Path, postprocess_mode: str, mode: str
) -> None:
    created: list[_FakeRunner] = []
    app = _make_app(tmp_path, created, load_asr=False)
    data = {
        "mode": mode,
        "text": "Hello there. How are you? Goodbye!",
        "insert_pauses": "true",
        "postprocess_mode": postprocess_mode,
        "speed": "1.2",
        "language": "en",
    }
    files = None
    if mode != "auto":
        data["instruct"] = "calm voice"
    if mode == "clone":
        data["ref_text"] = "Reference voice."
        files = {"ref_audio": ("ref.wav", _fake_wav_bytes(), "audio/wav")}
    with TestClient(app) as client:
        response = client.post("/generate", data=data, files=files)
    assert response.status_code == 200, response.text
    audio, rate = sf.read(BytesIO(response.content))
    assert rate == 24000
    assert len(audio) == 3 * 2400 + 2 * 7200
    np.testing.assert_array_equal(audio[2400:9600], 0)
    np.testing.assert_array_equal(audio[12000:19200], 0)
    assert np.all(audio[:2400] > 0)
    assert np.all(audio[9600:12000] > 0)
    assert np.all(audio[19200:] > 0)
    assert response.headers["x-omnivoice-audio-duration-s"] == "0.900"
    assert response.headers["x-omnivoice-sentence-count"] == "3"
    assert response.headers["x-omnivoice-insert-pauses"] == "true"
    assert response.headers["x-omnivoice-generate-total-ms"] == "33.00"
    assert (
        Path(response.headers["x-omnivoice-saved-path"]).read_bytes()
        == response.content
    )
    calls = created[0].model.calls
    assert [c["text"] for c in calls] == [
        ["Hello there."],
        ["How are you?"],
        ["Goodbye!"],
    ]
    for call in calls:
        assert call["speed"] == [1.2]
        assert call["language"] == ["en"]
        assert call["generation_config"].postprocess_mode == postprocess_mode
        if mode != "auto":
            assert call["instruct"] == ["calm voice"]
    if mode == "clone":
        assert len(created[0].model.prompt_calls) == 1
        assert all(
            c["voice_clone_prompt"][0] is calls[0]["voice_clone_prompt"][0]
            for c in calls
        )


@pytest.mark.parametrize("flag", [None, "false", "true"])
def test_insert_pauses_default_and_single_sentence(tmp_path: Path, flag) -> None:
    created: list[_FakeRunner] = []
    app = _make_app(tmp_path, created, load_asr=False)
    text = "One sentence." if flag == "true" else "First sentence. Second sentence."
    data = {"mode": "auto", "text": text}
    if flag is not None:
        data["insert_pauses"] = flag
    with TestClient(app) as client:
        response = client.post("/generate", data=data)
    assert response.status_code == 200
    audio, _ = sf.read(BytesIO(response.content))
    assert len(audio) == 2400
    assert created[0].model.calls[0]["text"] == [text]
    assert len(created[0].model.calls) == 1


def test_insert_pauses_validation(tmp_path: Path) -> None:
    created: list[_FakeRunner] = []
    app = _make_app(tmp_path, created, load_asr=False)
    with TestClient(app) as client:
        invalid = client.post(
            "/generate",
            data={"mode": "auto", "text": "Hello.", "insert_pauses": "maybe"},
        )
        duration = client.post(
            "/generate",
            data={
                "mode": "auto",
                "text": "Hello. Goodbye.",
                "insert_pauses": "true",
                "duration": "2",
            },
        )
    assert invalid.status_code == 422
    assert duration.status_code == 400
    assert "duration" in duration.json()["detail"]
    assert not created[0].model.calls


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Dr. Smith paid 3.14 dollars. Thanks!",
            ["Dr. Smith paid 3.14 dollars.", "Thanks!"],
        ),
        ('He said "Hello!" Then left.', ['He said "Hello!"', "Then left."]),
        ("你好。再见！", ["你好。", "再见！"]),
        ("Really?!\nYes... Okay", ["Really?!", "Yes...", "Okay"]),
        ("Visit example.com today. Bye.", ["Visit example.com today.", "Bye."]),
        (
            "Meet J. Smith in the U.S. today. Bye.",
            ["Meet J. Smith in the U.S. today.", "Bye."],
        ),
        ("No punctuation", ["No punctuation"]),
    ],
)
def test_split_sentences(text: str, expected: list[str]) -> None:
    from omnivoice_triton.serving.sentence_pauses import split_sentences

    assert split_sentences(text) == expected


def _design_request() -> dict[str, Any]:
    return {
        "data": {
            "mode": "design",
            "text": "Hello from the API.",
            "instruct": "female, young adult",
        }
    }


def test_clone_prompt_encoding_runs_on_the_batcher_thread(tmp_path: Path) -> None:
    """Prompt encoding must never overlap a CUDA Graph capture: it runs with batches."""
    created: list[_FakeRunner] = []
    app = _make_app(tmp_path, created, load_asr=False)

    with TestClient(app) as client:
        register = client.post(
            "/clone-prompts",
            data={"ref_text": "Registered transcript."},
            files={"ref_audio": ("ref.wav", _fake_wav_bytes(), "audio/wav")},
        )
        warm = client.post(
            "/cache/warm?wait=true",
            data={"mode": "clone", "ref_text": "Warmed transcript."},
            files={"ref_audio": ("ref.wav", _fake_wav_bytes(), "audio/wav")},
        )
        upload = client.post(
            "/generate",
            data={
                "mode": "clone",
                "text": "Clone me.",
                "ref_text": "Uploaded transcript.",
            },
            files={"ref_audio": ("ref.wav", _fake_wav_bytes(), "audio/wav")},
        )

    assert register.status_code == 200
    assert warm.status_code == 200
    assert upload.status_code == 200
    assert created[0].model.prompt_threads == ["omnivoice-batcher"] * 3
    assert created[0].model.prompt_grad_modes == [(False, True)] * 3


def test_generate_queue_overload_is_429_and_health_remains_ok(
    tmp_path: Path, monkeypatch
) -> None:
    created: list[_FakeRunner] = []
    app = _make_app(tmp_path, created, load_asr=False)
    with TestClient(app) as client:

        async def full(request):
            raise QueueFullError("Generation queue is full.")

        monkeypatch.setattr(app.state.batcher, "submit_async", full)
        response = client.post("/generate", **_design_request())
        assert response.status_code == 429
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["batcher_queue"]["pending_requests"] == 0
        assert health.json()["worker"]["active_requests"] == 0


def test_cuda_fault_in_batch_marks_worker_dead(tmp_path: Path) -> None:
    created: list[_FakeRunner] = []
    app = _make_app(tmp_path, created, load_asr=False, start_worker_callback=False)

    with TestClient(app) as client:
        created[0].model.generate_error = RuntimeError(
            "CUDA error: device-side assert triggered"
        )
        failed = client.post("/generate", **_design_request())
        assert failed.status_code == 503
        assert "device-side assert" in failed.json()["detail"]

        health = client.get("/health")
        assert health.status_code == 503
        payload = health.json()
        assert payload["status"] == "error"
        assert "device-side assert" in payload["cuda_fault"]["reason"]
        assert payload["worker"]["status"] == "dead"
        assert payload["worker"]["assignable"] is False

        healthz = client.get("/healthz")
        assert healthz.status_code == 503
        assert healthz.json()["error"] == "cuda_fault"
        assert client.get("/worker/state").json()["status"] == "dead"

        created[0].model.generate_error = None
        later = client.post("/generate", **_design_request())
        assert later.status_code == 503
        assert later.json()["error"] == "cuda_fault"
        register = client.post(
            "/clone-prompts",
            data={"ref_text": "Reference transcript."},
            files={"ref_audio": ("ref.wav", _fake_wav_bytes(), "audio/wav")},
        )
        assert register.status_code == 503
        assert register.json()["error"] == "cuda_fault"

    assert len(created[0].model.calls) == 1


def test_cuda_fault_in_prompt_encoding_marks_worker_dead(tmp_path: Path) -> None:
    created: list[_FakeRunner] = []
    app = _make_app(tmp_path, created, load_asr=False)

    with TestClient(app) as client:
        created[0].model.prompt_error = RuntimeError(
            "CUDA error: operation not permitted when stream is capturing"
        )
        register = client.post(
            "/clone-prompts",
            data={"ref_text": "Reference transcript."},
            files={"ref_audio": ("ref.wav", _fake_wav_bytes(), "audio/wav")},
        )
        assert register.status_code == 503
        assert client.get("/health").status_code == 503
        assert client.get("/healthz").status_code == 503


def test_cuda_out_of_memory_keeps_worker_healthy(tmp_path: Path) -> None:
    created: list[_FakeRunner] = []
    app = _make_app(tmp_path, created, load_asr=False)

    with TestClient(app) as client:
        created[0].model.generate_error = torch.OutOfMemoryError(
            "CUDA out of memory. Tried to allocate 20.00 MiB."
        )
        failed = client.post("/generate", **_design_request())
        assert failed.status_code == 500
        assert client.get("/health").status_code == 200
        assert client.get("/healthz").status_code == 200

        created[0].model.generate_error = None
        assert client.post("/generate", **_design_request()).status_code == 200


def test_failed_requests_clear_cuda_graphs_only_on_the_batcher_thread(
    tmp_path: Path,
) -> None:
    """Clearing graphs from a request thread could free one the batcher is replaying."""
    created: list[_FakeRunner] = []
    app = _make_app(tmp_path, created, load_asr=False)

    with TestClient(app) as client:
        model = created[0].model
        model.generate_error = ValueError("bad batch input")
        assert client.post("/generate", **_design_request()).status_code == 400
        model.generate_error = None

        model.prompt_error = RuntimeError("prompt encoder failed")
        register = client.post(
            "/clone-prompts",
            data={"ref_text": "Reference transcript."},
            files={"ref_audio": ("ref.wav", _fake_wav_bytes(), "audio/wav")},
        )
        assert register.status_code == 500
        assert client.get("/health").status_code == 200

    assert model.forward.clear_threads
    assert set(model.forward.clear_threads) == {"omnivoice-batcher"}
