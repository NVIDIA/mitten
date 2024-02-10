# MLPerf Inference Test Bench

**M**LPerf **I**nference **T**es**t** B**en**ch, or Mitten, is a framework by NVIDIA to run the [MLPerf Inference
benchmark](https://github.com/mlcommons/inference).

This is an in-progress refactoring and extending of the framework used in NVIDIA's
[MLPerf Inference v3.0](https://github.com/mlcommons/inference_results_v3.0/tree/main/closed/NVIDIA) and prior submissions.


## Features

Mitten, while more optimized for NVIDIA GPU-based systems, is a generic framework that supports arbitrary systems. Some
of the things that Mitten handles are:

- System hardware detection
- Describing and running a benchmark as a pipeline
- Building TRT engines from various sources

### Planned Features

- Executing C++ or other compiled executables as a pipeline operation inside Python
- Easier method of configuring pipelines
- Automatic debugging logs and artifacts
- Server-client system for benchmarking workloads over a network connection
