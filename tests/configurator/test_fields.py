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
import pytest
import argparse
import copy

from nvmitten.configurator.fields import *


def test_field_parse():
    f1 = Field("f1")
    args = parse_fields([f1], [])
    assert "f1" not in args
    assert len(args) == 0

    args = parse_fields([f1], ["--f1", "hello world"])
    assert "f1" in args
    assert args["f1"] == "hello world"
    assert len(args) == 1


def test_field_from_string():
    f1 = Field("f1", from_string=bool)
    f2 = Field("f2", from_string=int)
    f3 = Field("f3", from_string=lambda s: len(s))

    args = parse_fields([f1, f2, f3], ["--f2", "1337", "--f3", "123456"])
    assert "f1" not in args
    assert args["f2"] == 1337
    assert args["f3"] == 6

    args = parse_fields([f1, f2, f3], ["--f1"])
    assert len(args) == 1
    assert args["f1"] == True


def test_field_copy():
    f1 = Field("f1", from_string=bool)
    cp = copy.copy(f1)
    assert cp is f1


def test_field_deepcopy():
    f1 = Field("f1", from_string=bool)
    cp = copy.deepcopy(f1)
    assert cp is f1
