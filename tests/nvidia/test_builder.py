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


import numpy as np
import pytest
import tensorrt as trt

from nvmitten.nvidia.smi import NvSMI
if not NvSMI.check_functional():
    pytest.skip("Skipping GPU-only tests", allow_module_level=True)

from nvmitten.nvidia.builder import *
from nvmitten.nvidia.runner import EngineRunner
from nvmitten.pipeline import *
from nvmitten.nvidia.constants import TRT_LOGGER

class XORNetwork:
    def __init__(self):
        """Simple MLP to calculate XOR.
        """
        # Weights from a quick PyTorch run, then rounded
        self.hidden1 = np.array([[4.0, 6.0],
                                 [4.0, 6.0]], dtype=np.float32)

        self.bias1 = np.array([-6.0, -2.0], dtype=np.float32)

        self.hidden2 = np.array([[-9.0], [8.0]], dtype=np.float32)
        self.bias2 = np.array([-4.0], dtype=np.float32)

    @staticmethod
    def get_dataset():
        dataset = np.array([[0, 0, 0],
                            [0, 1, 1],
                            [1, 0, 1],
                            [1, 1, 0]], dtype=np.float32)

        X = np.ascontiguousarray(dataset[:,:-1])
        Y = np.ascontiguousarray(dataset[:,-1:])
        return X, Y

    def as_trt_network(self, network):
        x = network.add_input("input", trt.DataType.FLOAT, (-1, 2))
        xpand = network.add_shuffle(x)
        xpand.reshape_dims = (-1, 2)

        cons1_w = network.add_constant(trt.Dims([2, 2]), trt.Weights(self.hidden1))
        cons1_b = network.add_constant(trt.Dims([1, 2]), trt.Weights(self.bias1))
        h1 = network.add_matrix_multiply(xpand.get_output(0),
                                         trt.MatrixOperation.NONE,
                                         cons1_w.get_output(0),
                                         trt.MatrixOperation.NONE)
        h1_add_b = network.add_elementwise(h1.get_output(0), cons1_b.get_output(0), trt.ElementWiseOperation.SUM)
        s1 = network.add_activation(h1_add_b.get_output(0),
                                    trt.ActivationType.SIGMOID)

        cons2_w = network.add_constant(trt.Dims([2, 1]), trt.Weights(self.hidden2))
        cons2_b = network.add_constant(trt.Dims([1, 1]), trt.Weights(self.bias2))
        h2 = network.add_matrix_multiply(s1.get_output(0),
                                         trt.MatrixOperation.NONE,
                                         cons2_w.get_output(0),
                                         trt.MatrixOperation.NONE)
        h2_add_b = network.add_elementwise(h2.get_output(0), cons2_b.get_output(0), trt.ElementWiseOperation.SUM)
        s2 = network.add_activation(h2_add_b.get_output(0),
                                    trt.ActivationType.SIGMOID)

        network.mark_output(s2.get_output(0))
        return network


class XORBuilder(TRTBuilder, Operation):
    @classmethod
    def output_keys(cls):
        return ["batch_size", "engine_name", "engine"]

    @classmethod
    def immediate_dependencies(cls):
        return None

    def __init__(self,
                 *args,
                 engine_name: str = "xor_default.engine",
                 **kwargs):
        super().__init__(*args, **kwargs)

        self._network = XORNetwork()
        self.engine_name = engine_name

    def create_network(self, builder: trt.Builder = None) -> trt.INetworkDefinition:
        network = super().create_network(builder=builder)
        return self._network.as_trt_network(network)

    def run(self, scratch_space, dependency_outputs):
        batch_size = 4
        save_to = scratch_space.path / self.engine_name
        # save engine to /tmp
        self(batch_size, save_to)

        with open(save_to, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            buf = f.read()
            engine = runtime.deserialize_cuda_engine(buf)

        return {"batch_size": batch_size,
                "engine_name": self.engine_name,
                "engine": engine}


class XORRunner(Operation):
    @classmethod
    def output_keys(cls):
        return ["acc"]

    @classmethod
    def immediate_dependencies(cls):
        return [XORBuilder]

    def run(self, scratch_space, dependency_outputs):
        engine_name = scratch_space.path / dependency_outputs[XORBuilder]["engine_name"]
        assert Path(engine_name).exists()
        assert Path(engine_name).is_file()

        engine = dependency_outputs[XORBuilder]["engine"]
        assert engine is not None

        runner = EngineRunner(engine)
        x, y = XORNetwork.get_dataset()
        assert len(x) == len(y)

        batch_size = dependency_outputs[XORBuilder]["batch_size"]
        assert len(y) == batch_size

        outputs = runner([x], batch_size=batch_size)
        outputs = np.array(outputs).flatten()
        assert len(outputs) == batch_size

        acc = (outputs.round() == y.flatten()).sum() / batch_size
        return {"acc": acc}


@pytest.mark.skipif(not NvSMI.check_functional(), reason="Cannot run GPU tests when GPU is not available")
def test_builder_pipeline(tmp_path, debug_manager_io_stream):
    scratch_space = ScratchSpace(tmp_path)
    pipeline = Pipeline(scratch_space, [XORBuilder, XORRunner], dict())
    result = pipeline.run()
    assert result["acc"] == 1.0
