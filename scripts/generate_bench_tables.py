"""Read benchmark JSON results and generate markdown tables.

Reads benchmark/results/e2e_benchmarks.json and
benchmark/results/kernel_benchmarks.json, then either:
  - Replaces marked sections in README.md (if markers exist), or
  - Prints tables to stdout.

Marker comments in README.md:
    <!-- BENCH:E2E:START -->
    ... replaced content ...
    <!-- BENCH:E2E:END -->

    <!-- BENCH:KERNELS:START -->
    ... replaced content ...
    <!-- BENCH:KERNELS:END -->

Usage:
    uv run python scripts/generate_bench_tables.py
    uv run python scripts/generate_bench_tables.py --readme README.md
    uv run python scripts/generate_bench_tables.py --print-only
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("benchmark/results")
README_PATH = Path("README.md")

E2E_START = "<!-- BENCH:E2E:START -->"
E2E_END = "<!-- BENCH:E2E:END -->"
KERNEL_START = "<!-- BENCH:KERNELS:START -->"
KERNEL_END = "<!-- BENCH:KERNELS:END -->"


def _load_json(path: Path) -> list[Any] | dict[str, Any] | None:
    """Load JSON from path, returning None on failure."""
    if not path.exists():
        logger.warning("File not found: %s", path)
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse %s: %s", path, exc)
        return None


def _format_e2e_table(results: list[dict[str, Any]]) -> str:
    """Format E2E benchmark results as a markdown table.

    Args:
        results: List of benchmark result dicts from e2e_benchmarks.json.

    Returns:
        Markdown table string.
    """
    lines = [
        "| Runner | Lang | Mean (ms) | P50 (ms) | P95 (ms) | RTF | VRAM (GB) |",
        "|--------|------|----------:|---------:|---------:|----:|----------:|",
    ]
    for r in results:
        t = r.get("time_ms", {})
        rtf = r.get("rtf", {})
        lines.append(
            f"| {r.get('runner', '?')}"
            f" | {r.get('language', '?')}"
            f" | {t.get('mean', 0):.1f}"
            f" | {t.get('p50', 0):.1f}"
            f" | {t.get('p95', 0):.1f}"
            f" | {rtf.get('mean', 0):.2f}"
            f" | {r.get('peak_vram_gb', 0):.2f} |"
        )
    return "\n".join(lines)


def _format_kernel_table(results: list[dict[str, Any]]) -> str:
    """Format kernel benchmark results as a markdown table.

    Args:
        results: List of kernel benchmark dicts from kernel_benchmarks.json.

    Returns:
        Markdown table string.
    """
    lines = [
        "| Kernel | PyTorch (ms) | Triton (ms) | Speedup |",
        "|--------|-------------:|------------:|--------:|",
    ]
    for r in results:
        lines.append(
            f"| {r.get('kernel', '?')}"
            f" | {r.get('pytorch_ms', 0):.4f}"
            f" | {r.get('triton_ms', 0):.4f}"
            f" | {r.get('speedup', 0):.2f}x |"
        )
    return "\n".join(lines)


def _replace_between_markers(content: str, start: str, end: str, new_body: str) -> str:
    """Replace text between start/end markers (inclusive of markers).

    Args:
        content: Full file text to search.
        start: Start marker string.
        end: End marker string.
        new_body: Replacement text to insert between markers.

    Returns:
        Updated content string, or original if markers not found.
    """
    pattern = re.escape(start) + r".*?" + re.escape(end)
    replacement = f"{start}\n{new_body}\n{end}"
    updated, count = re.subn(pattern, replacement, content, flags=re.DOTALL)
    if count == 0:
        logger.debug("Markers not found in README: %s ... %s", start, end)
        return content
    return updated


def update_readme(
    readme_path: Path,
    e2e_table: str | None,
    kernel_table: str | None,
) -> bool:
    """Inject benchmark tables into README.md between marker comments.

    Args:
        readme_path: Path to README.md.
        e2e_table: Formatted E2E markdown table, or None to skip.
        kernel_table: Formatted kernel markdown table, or None to skip.

    Returns:
        True if README was modified, False otherwise.
    """
    if not readme_path.exists():
        logger.warning("README not found at %s", readme_path)
        return False

    content = readme_path.read_text(encoding="utf-8")
    original = content

    if e2e_table:
        content = _replace_between_markers(content, E2E_START, E2E_END, e2e_table)
    if kernel_table:
        content = _replace_between_markers(
            content, KERNEL_START, KERNEL_END, kernel_table
        )

    if content == original:
        return False

    readme_path.write_text(content, encoding="utf-8")
    return True


def main() -> None:
    """CLI entry point for generating benchmark tables."""
    parser = argparse.ArgumentParser(
        description="Generate markdown benchmark tables from JSON results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory containing benchmark JSON files.",
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=README_PATH,
        help="Path to README.md to update.",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print tables to stdout only; do not modify README.",
    )
    args = parser.parse_args()

    e2e_path = args.results_dir / "e2e_benchmarks.json"
    kernel_path = args.results_dir / "kernel_benchmarks.json"

    e2e_data = _load_json(e2e_path)
    kernel_data = _load_json(kernel_path)

    e2e_table: str | None = None
    kernel_table: str | None = None

    if e2e_data and isinstance(e2e_data, list):
        e2e_table = _format_e2e_table(e2e_data)
        logger.info("Generated E2E table (%d rows).", len(e2e_data))
    else:
        logger.warning("No E2E benchmark data available.")

    if kernel_data and isinstance(kernel_data, list):
        kernel_table = _format_kernel_table(kernel_data)
        logger.info("Generated kernel table (%d rows).", len(kernel_data))
    else:
        logger.warning("No kernel benchmark data available.")

    if args.print_only:
        if e2e_table:
            print("\n### E2E Benchmark Results\n")
            print(e2e_table)
        if kernel_table:
            print("\n### Kernel Benchmark Results\n")
            print(kernel_table)
        return

    modified = update_readme(args.readme, e2e_table, kernel_table)
    if modified:
        logger.info("README updated: %s", args.readme)
    else:
        logger.info(
            "README markers not found or no changes made. "
            "Add %s...%s and %s...%s to README.md, "
            "or use --print-only to view tables.",
            E2E_START,
            E2E_END,
            KERNEL_START,
            KERNEL_END,
        )
        # Always print to stdout as fallback
        if e2e_table:
            print("\n### E2E Benchmark Results\n")
            print(e2e_table)
        if kernel_table:
            print("\n### Kernel Benchmark Results\n")
            print(kernel_table)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)8s] %(message)s",
    )
    main()
