# Run documentation requirements

For every new or updated benchmark, experiment, validation report, notebook,
and run summary, state the exact GPU model near the beginning and per run/table
when hardware differs. Include physical/visible VRAM and the evidence source;
record driver/runtime, date, workload/profile, and co-resident load when known.

Distinguish GPU inference, CPU-only tests, static checks, reused historical
results, and external/upstream claims. Mark unknown hardware as unverified;
never infer a past run's GPU from the current machine alone. Allocator caps do
not change the physical GPU model. Recommended deployment GPUs are not tested
GPUs. Carry these labels into summaries and linked report indexes.
