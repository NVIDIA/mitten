# MLPerf Inference Test Bench

**M**LPerf **I**nference **T**es**t** B**en**ch, or Mitten, is a framework by NVIDIA to run the [MLPerf Inference
benchmark](https://github.com/mlcommons/inference).


## Features

Mitten, while more optimized for NVIDIA GPU-based systems, is a generic framework that supports arbitrary systems. Some
of the things that Mitten handles are:

- System hardware detection
- Describing and running a benchmark as a pipeline
- Building TRT engines from various sources
- Executing C++ or other compiled executables as a pipeline operation inside Python via pybind11
- Automatic debugging logs and artifacts


### Planned Features

- Easier method of configuring pipelines
- Server-client system for benchmarking workloads over a network connection
