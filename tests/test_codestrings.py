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

from dataclasses import dataclass
from enum import Enum, auto

import pytest

from nvmitten.codestrings import codestringable, obj_to_codestr


class ExampleEnum(Enum):
    Foo = auto()
    Bar = auto()
    Baz = auto()


@codestringable
class ExampleCodestringClass:
    def __init__(self, a, b, c, d=2, e="hello"):
        pass  # We don't really need to store anything


@codestringable
@dataclass
class ExampleCodestringDataclass:
    foo: str
    bar: int
    baz: bool = False


@pytest.mark.parametrize(
    "obj,expected_codestr",
    [
        (None, "None"),
        (ExampleEnum.Bar, "ExampleEnum.Bar"),
        ("hello", "\"hello\""),
        (23.7, "23.7"),
        (99, "99"),
        ({"foo": 1, "bar": "22"}, "{\"foo\": 1, \"bar\": \"22\"}"),
        (True, "True"),
        (ExampleCodestringClass(1, 2, 3, d=4, e="5"), "ExampleCodestringClass(1, 2, 3, d=4, e=\"5\")"),
        (ExampleCodestringDataclass("oof", 2, baz=True), "ExampleCodestringDataclass(\"oof\", 2, baz=True)"),
    ]
)
def test_obj_to_codestr(obj, expected_codestr):
    assert obj_to_codestr(obj) == expected_codestr
