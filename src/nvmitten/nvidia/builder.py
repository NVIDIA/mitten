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
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import copy
import numpy as np
import onnx
import onnx_graphsurgeon as gs
import re
import tensorrt as trt

from .constants import TRT_LOGGER, ComputeSM, WorkloadSetting
from .smi import NvSMI
from ..constants import Precision
from ..mlcommons.inference.constants import Benchmark, Scenario
from ..pipeline import ScratchSpace
from ..utils import logging


__doc__ = """This module contains NVIDIA implementations for MLPerf Inference Ops.
"""


_PotentiallyMultiple_T = Union[str, Dict, List, Tuple]


def get_dyn_ranges(cache_file: PathLike) -> Dict[str, np.uint32]:
    """Get dynamic ranges from calibration file for network tensors.

    Args:
        cache_file (PathLike): Path to INT8 calibration cache file.

    Returns:
        Dict[str, np.uint32]: Dictionary of tensor name -> dynamic range of tensor
    """
    cache_file = Path(cache_file)
    if not cache_file.exists():
        raise FileNotFoundError(f"Calibration cache {cache_file} does not exist.")

    with cache_file.open(mode="rb") as f:
        lines = f.read().decode('ascii').splitlines()

    dyn_ranges = {}
    for line in lines:
        regex = r"(.+): (\w+)"
        results = re.findall(regex, line)

        # Omit unmatched lines
        if len(results) == 0 or len(results[0]) != 2:
            continue

        results = results[0]
        tensor_name = results[0]
        # Map dynamic range from [0.0 - 1.0] to [0.0 - 127.0]
        dynamic_range = np.uint32(int(results[1], base=16)).view(np.dtype('float32')).item() * 127.0
        dyn_ranges[tensor_name] = dynamic_range
    return dyn_ranges


class ONNXNetwork:
    """Wrapper around tensorrt.INetworkDefinition and onnx-graphsurgeon.
    """

    @dataclass
    class Tensor:
        """Lightweight wrapper to denote a Tensor with a name and shape.
        """
        name: str
        shape: Tuple[Any, ...]

    @dataclass
    class Subnetwork:
        """Denotes a subnetwork of a graph by marking the inputs and outputs of the subnetwork.
        """
        inputs: List[Tensor] = field(default_factory=list)
        outputs: List[Tensor] = field(default_factory=list)

    def __init__(self,
                 onnx_path: PathLike,
                 precision: Precision,
                 num_inputs: int = 1,
                 calib_cache_path: Optional[PathLike] = None,
                 compute_sm: Optional[ComputeSM] = None,
                 op_name_remap: Optional[Dict[str, str]] = None,
                 no_fusions: bool = False):
        """Creates an ONNX-based tensorrt.INetworkDefinition wrapper.

        Args:
            onnx_path (PathLike): Path to the base ONNX file
            precision (Precision): The numeric precision to use for the network
            num_inputs (int): The number of inputs the ONNX graph should have. (Default: 1)
            calib_cache_path (PathLike): Path to the INT8 calibration cache, as exported by TensorRT. This is ignored if
                                         precision is not INT8. (Default: None)
            compute_sm (ComputeSM): The ComputeSM for the device the engine will be built for. This is useful for
                                    engines / networks that may want different fusions for different devices. If not
                                    set, this will be automatically detected from GPU 0. (Default: None)
            op_name_remap (Dict[str, str]): Defines what ONNX operations should be renamed in the final ONNX file. If
                                            not set, no ops will be renamed. (Default: None)
            no_fusions (bool): If True, self.fuse_ops() will not be called. (Default: False)
        """
        self.graph = self.import_onnx(onnx_path)

        self.precision = precision
        if precision == Precision.INT8 and calib_cache_path:
            self.dyn_range_map = get_dyn_ranges(calib_cache_path)
        else:
            self.dyn_range_map = dict()

        self.num_inputs = num_inputs

        if compute_sm:
            if isinstance(compute_sm, int):
                self.compute_sm = ComputeSM.from_int(compute_sm)
            else:
                self.compute_sm = compute_sm
        else:
            self.compute_sm = NvSMI.get_compute_sm(0)

        self.rename_nodes(op_name_remap)
        self.no_fusions = no_fusions

    def import_onnx(self, onnx_path: PathLike) -> gs.Graph:
        """Loads the ONNX file from a given path as an onnx-graphsurgeon Graph.

        Args:
            onnx_path (PathLike): The path of the ONNX file to load

        Returns:
            onnx_graphsurgeon.Graph: The ONNX as an onnx-gs Graph.
        """
        model = onnx.load(onnx_path)
        return gs.import_onnx(model)

    def rename_nodes(self, op_name_remap):
        """Renames the nodes of the graph in accordance to a remap dict, and then renames tensors to be consistent with
        their producer ops.
        """
        if not op_name_remap:
            return

        # Rename op names
        for node in self.graph.nodes:
            if node.name in op_name_remap:
                node.name = op_name_remap[node.name]

        # Rename tensors to be consistent wrt. their producer ops
        for node in self.graph.nodes:
            for t_idx, out_tensor in enumerate(node.outputs):
                if not out_tensor.name or not out_tensor.name.startswith(node.name):
                    out_tensor.name = f"{node.name}_out_{t_idx}"
        assert len(self.graph.inputs) == self.num_inputs, f"Expected {self.num_inputs} inputs. Got: {self.graph.inputs}"
        for input_ in self.graph.inputs:
            # It is assumed the input name is in the format "a_string:index" where 'index' is an int.
            input_.name = input_.name.replace(":", "_")

    def create_onnx_model(self, subnetwork: Subnetwork = None) -> onnx.ModelProto:
        """Creates the ONNX model. This method will:
        1. Perform a graph "touchup" where nodes can be replaced, added, or removed to match better with TRT Layers.
        2. Perform a graph fusion to combine nodes for TRT Layer Fusions or TRT Plugins.

        Args:
            subnetwork (Subnetwork): If set, will only return the marked subnetwork as an ONNX model. (Default: None)

        Returns:
            onnx.ModelProto: The ONNX model of the processed graph.
        """
        self.prefusion()
        if not self.no_fusions:
            self.fuse_ops()
        if subnetwork:
            # Set the inputs and outputs according to the desired subnetwork
            tensors = self.graph.tensors()
            if len(subnetwork.inputs) > 0:
                new_inputs = [tensors[tensor.name].to_variable(dtype=np.float32,
                                                               shape=tensor.shape)
                              for tensor in subnetwork.inputs]
                self.graph.inputs = new_inputs
            if len(subnetwork.outputs) > 0:
                new_outputs = [tensors[tensor.name].to_variable(dtype=np.float32,
                                                                shape=tensor.shape)
                               for tensor in subnetwork.outputs]
                self.graph.outputs = new_outputs
        self.cleanup_graph()
        return gs.export_onnx(self.graph)

    def cleanup_graph(self):
        self.graph.cleanup().toposort()

    def prefusion(self):
        """Implementation-defined function that is run before self.fuse_ops().
        """
        pass

    def fuse_ops(self):
        """Implementation-defined function on how to fuse ops.
        """
        pass


class TRTOptProfiles:
    """Wrapper around TRT IOptimizationProfile generators to only run once.
    """
    def __init__(self,
                 func: Callable[[trt.INetworkDefinition, int], List[trt.IOptimizationProfile]]):
        self.func = func
        self.has_run = False
        self.retval = None

    def __call__(self, network: trt.INetworkDefinition, batch_size: int) -> List[trt.IOptimizationProfile]:
        if not self.has_run:
            self.retval = self.func(network, batch_size)
            self.has_run = True
        return copy.copy(self.retval)  # Return shallow copy so cache isn't accidentally modified.


class TRTBuilder(ABC):
    """Sets up a TensorRT Builder. Used as a Mixin for an Operation.
    """

    def __init__(self,
                 *args,
                 precision: Precision = Precision.FP32,
                 input_dtype: _PotentiallyMultiple_T = "fp32",
                 input_format: _PotentiallyMultiple_T = "linear",
                 workspace_size: int = 1 << 30,
                 num_profiles: int = 1,
                 # TODO: Rename this to verbose_trt for clarity. --verbose will need to forwarded in Fields API.
                 verbose: bool = False,
                 verbose_nvtx: bool = False,
                 dla_core: int = None,
                 dla_sram: int = 1 << 20,
                 **kwargs):
        """Create a TensorRTBuilderOp.
        """
        super().__init__(*args, **kwargs)

        self.precision = precision

        # Handle input_dtype and input_format.
        assert type(input_dtype) is type(input_format), f"{self.__class__}.__init__: input_dtype and input_format " \
                                                        f"must be the same type."
        if isinstance(input_dtype, str):
            self.input_dtype = [input_dtype]
            self.input_format = [input_format]
            self.num_inputs = 1
        elif any(lambda _t: isinstance(input_dtype, _t), [tuple, list, dict]):
            self.input_dtype = copy.copy(input_dtype)
            self.input_format = copy.copy(input_format)
            assert len(self.input_dtype) == len(self.input_format), f"{self.__class__}.__init__: input_dtype and " \
                                                                    "input_format must be the same length."
            self.num_inputs = len(self.input_dtype)

        self.workspace_size = workspace_size
        self.num_profiles = num_profiles
        self.verbose = verbose
        self.verbose_nvtx = verbose_nvtx

        self.logger = TRT_LOGGER
        self.logger.min_severity = trt.Logger.VERBOSE if self.verbose else trt.Logger.INFO

        self.dla_enabled = dla_core is not None
        if self.dla_enabled:
            self.dla_core = int(dla_core)
            self.dla_sram = int(dla_sram)
            self.create_profiles = TRTOptProfiles(self.dla_profiles)
        else:
            self.create_profiles = TRTOptProfiles(self.gpu_profiles)

        trt.init_libnvinfer_plugins(self.logger, "")
        self.builder = trt.Builder(self.logger)

    def create_builder_config(self,
                              builder: trt.Builder = None,
                              workspace_size: Optional[int] = None,
                              profiles: Optional[List[trt.IOptimizationProfile]] = None,
                              precision: Optional[Precision] = None,
                              dla_core: Optional[int] = None,
                              dla_sram: Optional[int] = None,
                              verbose: Optional[bool] = None) -> trt.IBuilderConfig:
        """Creates a trt.IBuilderConfig from a builder.

        If a kwarg is left as None, the value will be extracted from `self`.

        Args:
            builder (trt.Builder): trt.Builder to create a config from.
            workspace_size (int): The size of the TensorRT workspace to create. (Default: None)
            profiles (List[trt.IOptimizationProfile]): List of optimization profiles to use. (Default: None)
            precision (Precision): The precision setting for the builder config. (Default: None)
            dla_core (int): The DLA core ID to use for the builder. If None, assumes DLA is not used, and no DLA
                            settings will be set. Otherwise, sets the default device to the specified DLA. (Default:
                            None)
            dla_sram (int): The DLA SRAM size to use. This setting is ignored if `dla_core` is None. (Default: None)
            verbose (bool): If True, will use detailed profiling verbosity. (Default: None)

        Returns:
            trt.IBuilderConfig: A TensorRT IBuilderConfig with the appropriate settings.
        """
        # Apply argument values from `self` if currently None
        if builder is None:
            builder = self.builder
        if workspace_size is None:
            workspace_size = self.workspace_size
        if precision is None:
            precision = self.precision
        if self.dla_enabled:
            if dla_core is None:
                dla_core = self.dla_core
            if dla_sram is None:
                dla_sram = self.dla_sram
        if verbose is None:
            verbose = self.verbose or self.verbose_nvtx

        # Create builder
        builder_config = builder.create_builder_config()
        builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
        if verbose:
            builder_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

        # Always disallow TF32 precision
        builder_config.clear_flag(trt.BuilderFlag.TF32)
        if precision == Precision.FP16:
            builder_config.set_flag(trt.BuilderFlag.FP16)
        elif precision == Precision.INT8:
            builder_config.set_flag(trt.BuilderFlag.INT8)

        # Apply DLA settings if applicable
        if dla_core is not None:
            builder_config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
            builder_config.default_device_type = trt.DeviceType.DLA
            builder_config.DLA_core = int(dla_core)

            # Currently with the TRT version in v3.0, TRT uses a 0.5MB DLA SRAM size by default,
            # which is suboptimal. Set the SRAM size to 1 MB.
            builder_config.set_memory_pool_limit(trt.MemoryPoolType.DLA_MANAGED_SRAM, dla_sram)

        # Set profiles if provided
        if profiles:
            for prof in profiles:
                builder_config.add_optimization_profile(prof)
        return builder_config

    @abstractmethod
    def create_network(self, builder: trt.Builder = None, flags: int = 0) -> trt.INetworkDefinition:
        if builder is None:
            builder = self.builder
        network = builder.create_network(flags)
        return network

    def build_engine(self,
                     network: trt.INetworkDefinition,
                     builder_config: trt.IBuilderConfig,
                     save_to: PathLike) -> trt.IHostMemory:
        # Build engines
        serialized_network = self.builder.build_serialized_network(network, builder_config)

        logging.debug("Saving engine")
        with save_to.open(mode='wb') as f:
            f.write(serialized_network)

        return serialized_network

    def __call__(self,
                 batch_size: int,
                 save_to: PathLike,
                 network: Optional[trt.INetworkDefinition] = None,
                 profiles: Optional[List[trt.IOptimizationProfile]] = None,
                 builder_config: Optional[trt.IBuilderConfig] = None) -> trt.IHostMemory:
        if network is None:
            network = self.create_network()

        if profiles is None:
            if network.has_implicit_batch_dimension:
                logging.info(f"Network uses implicit batch size. Setting max_batch_size to {batch_size}.")
                self.builder.max_batch_size = batch_size
                profiles = list()
            else:
                logging.info(f"Building optimization profiles.")
                profiles = self.create_profiles(network, batch_size)

        if builder_config is None:
            builder_config = self.create_builder_config(profiles=profiles)

        save_to = Path(save_to)
        if save_to.is_file():
            logging.warning(f"{save_to} already exists. This file will be overwritten")
        save_to.parent.mkdir(parents=True, exist_ok=True)

        logging.info(f"Building engine to {save_to}")
        return self.build_engine(network, builder_config, save_to)

    def gpu_profiles(self, network: trt.INetworkDefinition, batch_size: int):
        profiles = []

        for i in range(self.num_profiles):
            if i < len(profiles):
                logging.info(f"Reusing profile: {i}")
                profile = profiles[i]
            else:
                profile = self.builder.create_optimization_profile()

            # Set profile input shapes
            for input_idx in range(network.num_inputs):
                input_shape = network.get_input(input_idx).shape
                input_name = network.get_input(input_idx).name
                min_shape = trt.Dims(input_shape)
                min_shape[0] = 1
                max_shape = trt.Dims(input_shape)
                max_shape[0] = batch_size
                profile.set_shape(input_name, min_shape, max_shape, max_shape)

            if not profile:
                raise RuntimeError("Invalid optimization profile!")

            if i >= len(profiles):
                profiles.append(profile)
        return profiles

    def dla_profiles(self, network: trt.INetworkDefinition, batch_size: int):
        # Use fixed batch size for DLA
        for input_idx in range(network.num_inputs):
            input_shape = network.get_input(input_idx).shape
            input_shape[0] = batch_size
            network.get_input(input_idx).shape = input_shape

        # Signal that we don't want to call IBuilderConfig.add_optimization_profile.
        return []


class CalibratableTensorRTEngine:
    """Used as a Mixin for a TRTBuilder that supports INT8 calibration.
    """

    def __init__(self,
                 *args,
                 calib_batch_size: int = 1,
                 calib_max_batches: int = 500,
                 force_calibration: bool = False,
                 calib_data_map: PathLike = None,
                 cache_file: PathLike = None,
                 **kwargs):
        """Create a CalibratableTensorRTEngine. Must be used in conjunction with TRTBuilder.
        """
        super().__init__(*args, **kwargs)

        self.calib_batch_size = calib_batch_size
        self.calib_max_batches = calib_max_batches
        self.force_calibration = force_calibration
        self.calib_data_map = Path(calib_data_map)
        self.cache_file = Path(cache_file)

    @property
    def need_calibration(self):
        return self.force_calibration or not self.cache_file.exists()

    def get_calibrator(self, image_dir: PathLike) -> trt.IInt8Calibrator:
        # Implementation must implement a trt.IInt8Calibrator and return an instance of it in this method.
        return None

    def calibration_profiles(self, network: trt.INetworkDefinition, batch_size: int):
        for input_idx in range(network.num_inputs):
            input_shape = network.get_input(input_idx).shape
            input_shape[0] = self.calib_batch_size
            network.get_input(input_idx).shape = input_shape


class MLPerfInferenceEngine:
    """Used as a Mixin for an TRTBuilder that is also for an MLPerf Inference Benchmark.
    """

    def __init__(self,
                 *args,
                 system: None,
                 benchmark: None,
                 scenario: None,
                 **kwargs):
        """Create a TensorRTBuilderOp.
        """
        super().__init__(*args, **kwargs)

        assert system is not None, "Must specify the system to build for"
        assert benchmark is not None, "Must specify the benchmark (model name) for the engine"
        assert scenario is not None, "Must specify the scenario for the engine"
        self.system = system
        self.benchmark = benchmark
        self.scenario = scenario

    def engine_dir(self, scratch_space: ScratchSpace) -> Path:
        system_id = self.system.extras["id"]
        name = self.benchmark.valstr()
        scenario = self.scenario.valstr()
        return scratch_space.path / "engines" / system_id / name / scenario

    def engine_name(self,
                    device_type: str,
                    batch_size: int,
                    precision: str,
                    subnetwork_name: str = None,
                    tag: str = "default") -> str:
        """Gets the name of the engine, constructed from the device it is build for, the explicit batch size, and an
        optional tag.

        Args:
            device_type (str): The device that TRT is building the engine for. Either "gpu" or "dla".
            batch_size (int): The max batch size / explicit batch size the engine is built with.
            precision (str): The lowest precision enabled for the engine.
            tag (str): A tag to use for the engine. (Default: "default")
            subnetwork_name (str): If applicable, name of the subnetwork the engine is built for. (Default: None)

        Returns:
            str: The name of the engine.
        """
        if not precision:
            if hasattr(self, "precision"):
                # TODO: self.precision is currently a string, but in the case it is an AliasedNameEnum member,
                # add support to use .valstr()
                precision = self.precision
            else:
                raise ValueError("precision cannot be None if self.precision is not set.")

        if subnetwork_name:
            device_type = f"{device_type}-{subnetwork_name}"

        name = self.benchmark.valstr()
        scenario = self.scenario.valstr()
        return f"{name}-{scenario}-{device_type}-b{batch_size}-{precision}.{tag}.plan"


class LegacyBuilder:
    """Wraps around Mitten TensorRT EngineBuilder API to be compatible with the old EngineBuilder API from NVIDIA's v3.0
    and prior submissions.

    DeprecationWarning: This class will be removed when the main submission codebase is fully refactored.
    """
    def __init__(self, mitten_builder):
        self.legacy_scratch = ScratchSpace("build")
        self.mitten_builder = mitten_builder

    def _get_engine_fpath(self, device_type, batch_size):
        # TODO: This method only exists to maintain compatibility with the old API while the main submission codebase is
        # being refactored. Remove this once the refactoring is complete.
        # Note that calling this method outside of the old API may error.
        engine_dir = self.mitten_builder.engine_dir(self.legacy_scratch)
        engine_name = self.mitten_builder.engine_name(device_type,
                                                      batch_size,
                                                      self.mitten_builder.precision,
                                                      tag=self.mitten_builder.config_ver)
        return engine_dir / engine_name

    def build_engines(self):
        # Do not run the full Mitten pipeline yet. Invoke run manually.
        self.mitten_builder.run(self.legacy_scratch, None)

    # TODO: this function is deprecated and should be removed as the body of it has been moved to MLPerf code base
    def calibrate(self):
        # Do nothing if the builder isn't a CalibratableTensorRTEngine.
        if not isinstance(self.mitten_builder, CalibratableTensorRTEngine):
            if self.mitten_builder.verbose:
                logging.info("Builder is not instance of CalibratableTensorRTEngine. Skipping calibrate.")
            return

        old_fields = dict()

        def _cache_and_set(attr, val):
            old_fields[attr] = getattr(self.mitten_builder, attr)
            setattr(self.mitten_builder, attr, val)

        # Unlike the old legacy API, we don't need to call .clear_cache(), since the calibrator is created in
        # TRTBuilder.run() instead of __init__.
        # Note that after calibration, TensorRT will still build the engine. In this case, we set the batch size to 1 to
        # make it go faster, but I'm not sure how to skip it.
        _cache_and_set("force_calibration", True)
        _cache_and_set("batch_size", 1)
        _cache_and_set("create_profiles", self.mitten_builder.calibration_profiles)

        # Do not run the full Mitten pipeline yet. Invoke run manually.
        self.mitten_builder.run(self.legacy_scratch, None)

        # Restore old values
        for attr, val in old_fields.items():
            setattr(self.mitten_builder, attr, val)
