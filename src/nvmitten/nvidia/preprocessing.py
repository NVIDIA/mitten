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


__doc__ = """This module contains NVIDIA implementations for data preprocessing. These implementations are designed to
work with NVIDIA's MLPerf Inference submission, and are not guaranteed to work for other implementations.
"""


from ..mlcommons.inference.datasets import *
from ..pipeline import Operation


class KiTS19PreprocessOp(Operation):

    @classmethod
    def immediate_dependencies(cls):
        return {KiTS19DatasetOp}

    def run(self, scratch_space, dependency_outputs):
        # TODO
        return True


class SQuADv1PreprocessOp(Operation):

    @classmethod
    def immediate_dependencies(cls):
        return {SQuADv1_1DatasetOp}

    def run(self, scratch_space, dependency_outputs):
        # TODO
        return True


class CriteoPreprocessOp(Operation):

    @classmethod
    def immediate_dependencies(cls):
        return {Criteo1TBDatasetOp}

    def run(self, scratch_space, dependency_outputs):
        # TODO
        return True


class ImagenetPreprocessOp(Operation):

    @classmethod
    def immediate_dependencies(cls):
        return {ILSVRC2012DatasetOp}

    def run(self, scratch_space, dependency_outputs):
        # TODO
        return True


class OpenImagesPreprocessOp(Operation):

    @classmethod
    def immediate_dependencies(cls):
        return {OpenImagesV6MLPerfOp}

    def run(self, scratch_space, dependency_outputs):
        # TODO
        return True


class LibriSpeechPreprocessOp(Operation):

    @classmethod
    def immediate_dependencies(cls):
        return {LibriSpeechDatasetOp}

    def run(self, scratch_space, dependency_outputs):
        # TODO
        return True
