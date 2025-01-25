# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
"""Wrapper around cuda-python to extend some functionality and make API more Pythonic.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from cuda import cuda, cudart, nvrtc
from types import SimpleNamespace
from typing import Callable, Iterable, Optional

import inspect
import logging
import numpy as np
import uuid


class _ModuleWrapper:
    """Wraps around each of the submodules of cuda-python (cuda.cuda, cuda.cudart, cuda.nvrtc) and makes method
    interfaces more Pythonic.

    Major changes are to return only the output if the methods succeed, and raise Python errors otherwise.
    """

    def __init__(self, mod, err_enum, err_desc_fn):
        """Automatically populates a subclass of _ModuleWrapper with the following members:
            - Error: A class with member classes representing all the errors derived from the error codes in this base
              module.
            - Using __getattr__ on a callable from the base module will instead return a wrapped function which raises
              non-Success errors, and returns only the return values without error codes.
        """

        self._base_mod = mod

        # Generate way to directly access exception classes via a self.Error member (i.e.
        # cuda.Error.CUDA_ERROR_OUT_OF_MEMORY)
        # This method behaves similar to an Enum, except members are the class themselves, not Enum objects.
        #
        # >>> issubclass(cuda.Error.CUDA_ERROR_DEVICE_UNAVAILABLE, cuda.Error)
        # True
        class _Error_T(Exception):
            _err_code_map = dict()

            @classmethod
            def __init_subclass__(cls, base, err_desc_fn):
                setattr(_Error_T, cls.__name__, cls)
                _Error_T._err_code_map[base] = cls

                cls.base_enum_member = base
                cls.err_msg = err_desc_fn(base)[1]

            def __init__(self):
                _cls = self.__class__
                super().__init__(f"err_no={_cls.base_enum_member.value}, desc={_cls.err_msg}")

            @classmethod
            def from_errcode(cls, err):
                return _Error_T._err_code_map[err]

        self.Error = _Error_T

        # Dynamically create Python exception types for all error codes
        for err in err_enum:
            if err.value == 0:  # 0 always represents success.
                self._success_code = err
            else:
                # Dynamically generate class definition
                _ = type(err.name, (self.Error,), dict(), base=err, err_desc_fn=err_desc_fn)

    def __getattr__(self, name):
        if not hasattr(self._base_mod, name):
            raise AttributeError

        attr = getattr(self._base_mod, name)

        if callable(attr) and not inspect.isclass(attr):
            # cuda-python methods always return an error code, and then return values if any. We want to instead return
            # the output, but raise the error if the error code is not success
            def _f(*args, **kwargs):
                retval = attr(*args, **kwargs)

                assert type(retval) is tuple
                assert len(retval) > 0

                err = retval[0]
                if err != self._success_code:
                    raise self.Error.from_errcode(err)

                if len(retval) == 1:
                    return None
                elif len(retval) == 2:
                    return retval[1]
                else:
                    return retval[1:]

            return _f

        else:
            return attr


CUDAWrapper = _ModuleWrapper(cuda,
                             cuda.CUresult,
                             cuda.cuGetErrorString)

CUDARTWrapper = _ModuleWrapper(cudart,
                               cudart.cudaError_t,
                               cudart.cudaGetErrorString)

NVRTCWrapper = _ModuleWrapper(nvrtc,
                              nvrtc.nvrtcResult,
                              nvrtc.nvrtcGetErrorString)


class CuPyContextManager:
    """Makes CUDA contexts more Pythonic, allowing them to be used as scoped contexts.
    """

    def __init__(self, device_id: int = 0, flags: int = 0):
        CUDAWrapper.cuInit(0)
        self.device = CUDAWrapper.cuDeviceGet(device_id)
        self.flags = flags
        self._context = None

    def __enter__(self):
        assert self._context is None
        self._context = CUDAWrapper.cuCtxCreate(self.flags, self.device)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        CUDAWrapper.cuCtxDestroy(self._context)


def np_from_pointer(pointer: int,
                    shape: Iterable[int],
                    nptype: type,
                    copy: bool = False,
                    read_only: bool = False) -> np.ndarray:
    """Source: from https://stackoverflow.com/a/56755422

    Generates numpy array from memory address
    https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.interface.html

    Args:
        pointer (int): Memory address
        shape (Iterable[int]): Shape of array.
        nptype (type): Numpy datatype for tensor elements
        copy (bool): Copy memory when constructing array. If False, the returned Numpy array will use the same
                     underlying memory that `pointer` points to. (Default: False)
        read_only (bool): Create the numpy array as read only array. (Default: False)

    Returns:
        np.ndarray: Numpy array with data from pointer.
    """

    np_to_typestr = {
        # https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.interface.html#__array_interface__
        # Example: little-endian FP16 = '<f2'
        np.float32: "<f4",
        np.float16: "<f2",
        np.int32: "<i4",
    }

    assert nptype in np_to_typestr, "Format not yet supported!"
    buff = {
        'data': (pointer, read_only),
        'typestr': np_to_typestr[nptype],
        'shape': tuple(shape),
        'version': 3,
    }

    class numpy_holder():
        pass

    holder = numpy_holder()
    holder.__array_interface__ = buff
    return np.array(holder, copy=copy)


@dataclass
class HostDeviceBuffer:
    """Represents a unified Host and Device memory buffer. The user is responsible for calling `h2d()` or `h2d()`.

    In cases where allocating a host or device buffer automatically is not desired, passing in `0` as the value for
    `host` or `device` will disable that buffer from being automatically created.
    """
    shape: Tuple[int, ...]
    nptype: type = np.float32
    name: str = ""
    host: int = None
    device: int = None
    _owns_host: bool = field(init=False)
    _owns_device: bool = field(init=False)
    _bytesize: int = field(init=False)

    def __post_init__(self):
        # Set name to random UUID if not provided
        if len(self.name) == 0:
            self.name = str(uuid.uuid4())

        # Calculate bytesize
        if not isinstance(self.shape, tuple):
            self.shape = tuple(self.shape)
        _vol = np.prod(self.shape)
        self._bytesize = _vol * np.dtype(self.nptype).itemsize

        # Instantiate CUDA buffers if not provided
        if self.host is None:
            self.host = CUDARTWrapper.cudaMallocHost(self._bytesize)
            self._owns_host = True
        else:
            self._owns_host = False

        if self.device is None:
            self.device = CUDARTWrapper.cudaMalloc(self._bytesize)
            self._owns_device = True
        else:
            self._owns_device = False

    def __del__(self):
        if self.host is not None and self._owns_host:
            logging.debug(f"Freeing host {self.host} for {self.name}")
            CUDARTWrapper.cudaFreeHost(self.host)

        if self.device is not None and self._owns_device:
            logging.debug(f"Freeing device {self.device} for {self.name}")
            CUDARTWrapper.cudaFree(self.device)

    def _check_host(fn: Callable):
        def _f(inst, *args, **kwargs):
            if inst.host is None or inst.host == 0:
                raise BufferError(f"Host buffer for {inst.name} is not allocated.")

            return fn(inst, *args, **kwargs)
        return _f

    def _check_device(fn: Callable):
        def _f(inst, *args, **kwargs):
            if inst.device is None or inst.device == 0:
                raise BufferError(f"Device buffer for {inst.name} is not allocated.")

            return fn(inst, *args, **kwargs)
        return _f

    @_check_device
    def h2d_async(self,
                  stream: cudart.cudaStream_t,
                  host: Optional[int] = None):
        if host is None and (self.host is None or self.host == 0):
            raise BufferError(f"Host buffer for {self.name} is not allocated.")
        CUDARTWrapper.cudaMemcpyAsync(self.device,
                                      host if host is not None else self.host,
                                      self._bytesize,
                                      CUDARTWrapper.cudaMemcpyKind.cudaMemcpyHostToDevice,
                                      stream)

    @_check_host
    @_check_device
    def d2h_async(self,
                  stream: cudart.cudaStream_t):
        CUDARTWrapper.cudaMemcpyAsync(self.host,
                                      self.device,
                                      self._bytesize,
                                      CUDARTWrapper.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                                      stream)

    @_check_host
    def host_to_ndarray(self,
                        copy: bool = True,
                        read_only: bool = False) -> np.ndarray:
        return np_from_pointer(self.host,
                               self.shape,
                               self.nptype,
                               copy=copy,
                               read_only=read_only)

    @_check_host
    def ndarray_to_host(self,
                        a: np.ndarray,
                        stream: cudart.cudaStream_t,
                        sync: bool = False):
        assert a.dtype == np.dtype(self.nptype)

        CUDARTWrapper.cudaMemcpyAsync(self.host,
                                      a.tobytes(),
                                      self._bytesize,
                                      CUDARTWrapper.cudaMemcpyKind.cudaMemcpyHostToHost,
                                      stream)

        if sync:
            CUDARTWrapper.cudaStreamSynchronize(stream)


class ScopedCUDAStream:
    def __init__(self):
        self._stream = None

    def __enter__(self):
        self._stream = CUDARTWrapper.cudaStreamCreate()
        return self._stream

    def __exit__(self, type, value, traceback):
        CUDARTWrapper.cudaStreamDestroy(self._stream)
