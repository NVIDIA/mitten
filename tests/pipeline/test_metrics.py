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

import pytest

from nvmitten.pipeline.metrics import *


@pytest.mark.parametrize(
    "b1,b2,expected",
    [
        (BenchmarkMetric("img/s"), BenchmarkMetric("images per second"), False),
        (BenchmarkMetric("img/s"), BenchmarkMetric("img/s"), True),
        (BenchmarkMetric("img/s", bigger_is_better=False), BenchmarkMetric("images per second"), False),
        (BenchmarkMetric("img/s"), BenchmarkMetric("img/s", bigger_is_better=False), False)
    ]
)
def test_benchmark_metric(b1, b2, expected):
    # TODO: BenchmarkMetric usage isn't well defined yet. This test isn't made to be too robust, but is just for
    # coverage.
    assert (b1 == b2) is expected
