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

from pathlib import Path

import pytest
import shutil

from nvmitten.pipeline.scratch_space import *


@pytest.mark.parametrize(
    "ssroot,namespace",
    [
        ("/tmp/nvmitten/test/scratch_space", "a1"),
        ("/tmp/nvmitten/test/scratch_space", "a2"),
        ("/tmp/nvmitten/test/scratch_space", "a3"),
        ("/tmp/nvmitten/test/scratch_space", ""),
    ]
)
def test_scratch_space(ssroot, namespace):
    # This test directly modifies the filesystem, so there is a small chance the test sporadically reports a false
    # positive due to a directory existing due to external reasons, or a false negative if the directory was deleted by
    # external reasons.
    p = Path(ssroot)
    if p.exists():
        shutil.rmtree(p)
    assert not p.exists()

    ss = ScratchSpace(ssroot)
    assert not p.exists()
    ss.create()
    assert p.exists()
    wd = ss.working_dir(namespace=namespace)
    assert wd.exists()
