# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations
from dataclasses import dataclass
from packaging import version
from pathlib import Path
from typing import List, Tuple

import ctypes
import logging
import numpy as np
import os
import tensorrt as trt

from .constants import TRT_LOGGER
from .cupy import (
    CUDAWrapper as cuda,
    CUDARTWrapper as cudart,
    HostDeviceBuffer,
)

# Numpy 1.24.0+ has removed np.bool as a datatype. TensorRT 8.6.0.6 (MLPINF v3.0 release version) has not captured this,
# but was fixed in 8.6.1.
if version.parse(trt.__version__) < version.parse("8.6.1"):
    typemap = {
        trt.float32: np.float32,
        trt.float16: np.float16,
        trt.int8: np.int8,
        trt.int32: np.int32,
        trt.bool: np.bool_,
        trt.uint8: np.uint8,
    }
    nptype = typemap.get
else:
    nptype = trt.nptype


def allocate_buffers(engine: trt.ICudaEngine,
                     profile_id: int):
    """
    Allocate device memory based on the engine tensor I/O.
    Use TensorRT 8.6 API instead of the deprecated binding syntax.

    Returns:
        input_tensors (dict[str, tuple[int, int]]): Mapping from name to (ptr, bytesize) tuples for input tensor
        outputs (List[HostDeviceBuffer]): output host-device address pair
    """
    num_io_tensors = engine.num_io_tensors
    input_tensors = {}

    d_inputs, outputs, bindings = [], [], []
    if engine.has_implicit_batch_dimension:
        max_batch_size = engine.max_batch_size
    else:
        # NOTE: We are assuming that input idx 0 has dynamic shape, and the only dynamic dim is the batch dim,
        # which might not be true.
        tensor_name = engine.get_tensor_name(0)
        shape = engine.get_tensor_shape(tensor_name)
        if -1 in list(shape):
            batch_dim = list(shape).index(-1)
            _prof_shape = engine.get_tensor_profile_shape(tensor_name, profile_id)
            _max_dims = _prof_shape[2]  # [min, opt, max]
            max_batch_size = _max_dims[batch_dim]
        else:
            max_batch_size = shape[0]

    for tensor_idx in range(num_io_tensors):
        tensor_name = engine.get_tensor_name(tensor_idx)
        tensor_io_type = engine.get_tensor_mode(tensor_name)
        tensor_dtype = engine.get_tensor_dtype(tensor_name)
        tensor_format = engine.get_tensor_format(tensor_name)
        tensor_shape = engine.get_tensor_shape(tensor_name)
        logging.info(
            f"Tensor idx: {tensor_idx}, Tensor name: {tensor_name}, "
            f"iotype: {tensor_io_type}, dtype: {tensor_dtype}, "
            f"format: {tensor_format}, shape: {tensor_shape}"
        )

        if tensor_format in [trt.TensorFormat.CHW4, trt.TensorFormat.DLA_HWC4]:
            tensor_shape[-3] = ((tensor_shape[-3] - 1) // 4 + 1) * 4
        elif tensor_format in [trt.TensorFormat.CHW32]:
            tensor_shape[-3] = ((tensor_shape[-3] - 1) // 32 + 1) * 32
        elif tensor_format == trt.TensorFormat.DHWC8:
            tensor_shape[-4] = ((tensor_shape[-4] - 1) // 8 + 1) * 8
        elif tensor_format == trt.TensorFormat.CDHW32:
            tensor_shape[-4] = ((tensor_shape[-4] - 1) // 32 + 1) * 32
        if not engine.has_implicit_batch_dimension:
            if -1 in list(tensor_shape):
                batch_dim = list(tensor_shape).index(-1)
                tensor_shape[batch_dim] = max_batch_size
        else:
            tensor_shape = (max_batch_size, *tensor_shape)

        # Allocate device buffers
        is_input = (tensor_io_type == trt.TensorIOMode.INPUT)
        _buff = HostDeviceBuffer(tensor_shape,
                                 nptype(tensor_dtype),
                                 name=tensor_name,
                                 host=(0 if is_input else None))

        # Append to the appropriate list.
        if is_input:
            input_tensors[tensor_name] = _buff
        elif tensor_io_type == trt.TensorIOMode.OUTPUT:
            outputs.append(_buff)
    return input_tensors, outputs


class EngineRunner:
    """Enable running inference through an engine on each call."""

    def __init__(self,
                 engine: trt.ICudaEngine,
                 verbose: bool = False,
                 profile_id: int = 0):
        """Load engine from file, allocate device memory for its bindings and create execution context."""

        self.engine = engine
        self.verbose = verbose
        TRT_LOGGER.min_severity = trt.Logger.VERBOSE if self.verbose else trt.Logger.INFO
        if profile_id < 0:
            profile_id = self.engine.num_optimization_profiles + profile_id

        self.input_tensor_map, self.outputs = allocate_buffers(self.engine, profile_id)
        self.stream = cudart.cudaStreamCreate()
        self.context = self.engine.create_execution_context()

        # Set context tensor address (equivalent to set binding in old syntax)
        for i in self.input_tensor_map.values():
            self.context.set_tensor_address(i.name, int(i.device))
        for o in self.outputs:
            self.context.set_tensor_address(o.name, int(o.device))

        if profile_id > 0:
            self.context.active_optimization_profile = profile_id

    @classmethod
    def from_file(cls,
                  engine_file: os.PathLike,
                  plugins: List[str] = None,
                  **kwargs) -> EngineRunner:
        engine_file = Path(engine_file)
        engine_file.is_file()  # Implicitly throws FileNotFoundError

        trt.init_libnvinfer_plugins(TRT_LOGGER, "")
        if plugins is not None:
            for plugin in plugins:
                ctypes.CDLL(plugin)

        engine = cls.load_engine(engine_file)
        return cls(engine, **kwargs)

    @classmethod
    def load_engine(cls, src_path: os.PathLike) -> trt.ICudaEngine:
        """Deserialize engine file to an engine and return it."""
        with open(src_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            buf = f.read()
            engine = runtime.deserialize_cuda_engine(buf)
        return engine

    def __call__(self,
                 inputs: np.ndarray,
                 batch_size: int = 1,
                 copy_outputs: bool = True,
                 read_only_outputs: bool = False) -> List[np.ndarray]:
        """Use host inputs to run inference on device and return back results to host."""

        # Copy input data to device bindings of context.
        assert len(inputs) == len(self.input_tensor_map), \
            f"Feeding {len(inputs)} inputs, but the engine only has {len(self.input_tensor_map)} allocations."
        input_buffers = [self.input_tensor_map[self.engine.get_tensor_name(idx)] for idx in range(len(inputs))]

        [buff.h2d_async(self.stream, host=inp)
         for (buff, inp) in zip(input_buffers, inputs)]

        # Run inference.
        if self.engine.has_implicit_batch_dimension:
            self.context.execute_async(batch_size=batch_size, bindings=self.bindings, stream_handle=self.stream)
        else:
            for tensor_idx in range(self.engine.num_io_tensors):
                tensor_name = self.engine.get_tensor_name(tensor_idx)
                if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                    input_shape = self.engine.get_tensor_shape(tensor_name)
                    if -1 in list(input_shape):
                        input_shape[0] = batch_size
                        self.context.set_input_shape(tensor_name, input_shape)
            self.context.execute_async_v3(self.stream)

        # Copy output device buffers back to host.
        [
            out.d2h_async(self.stream)
            for out in self.outputs
        ]

        # Synchronize the stream
        cudart.cudaStreamSynchronize(self.stream)

        # Return only the host outputs.
        return [out.host_to_ndarray(copy=copy_outputs,
                                    read_only=read_only_outputs)
                for out in self.outputs]

    def __del__(self):
        # Clean up everything.
        with self.engine, self.context:
            for buff in self.input_tensor_map.items():
                del buff
            for buff in self.outputs:
                del buff
            cudart.cudaStreamDestroy(self.stream)
