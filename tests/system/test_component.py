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
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, ClassVar, List

import pytest

from nvmitten.aliased_name import AliasedName
from nvmitten.interval import NumericRange
from nvmitten.json_utils import loads, dumps
from nvmitten.system.component import *


@dataclass
class DummyComponent(Component):
    str_field: str
    float_field: float
    list_field: List

    times_called: ClassVar[int] = 0

    @classmethod
    def detect(cls) -> List[DummyComponent]:
        DummyComponent.times_called += 1
        return [DummyComponent("foo", 123.45, ['a', 'b', 3]),
                DummyComponent("bar", 1337.42, [None])]


@dataclass
class DummySystem(Component):
    subcomponents: List[DummyComponent]

    @classmethod
    def detect(cls) -> List[DummySystem]:
        return [DummySystem(DummyComponent.detect())]


def test_component():
    DummyComponent.detect.cache_clear()
    DummyComponent.times_called = 0

    comps = DummyComponent.detect()

    assert len(comps) == 2
    assert comps[0].str_field == "foo"
    assert comps[0].float_field == 123.45
    assert comps[0].list_field == ['a', 'b', 3]

    assert comps[1].str_field == "bar"
    assert comps[1].float_field == 1337.42
    assert comps[1].list_field == [None]


def test_component_detect_cache():
    DummyComponent.detect.cache_clear()
    DummyComponent.times_called = 0

    comps1 = DummyComponent.detect()
    assert DummyComponent.times_called == 1

    comps2 = DummyComponent.detect()
    assert DummyComponent.times_called == 1  # Should be cache hit

    assert comps1 == comps2

    ci = DummyComponent.detect.cache_info()
    assert ci.hits == 1
    assert ci.misses == 1


def test_component_bad_definition():
    class BrokenComponent(Component):
        pass

    with pytest.raises(AssertionError):
        Description(BrokenComponent)


@pytest.mark.parametrize(
    "idx,desc,expected",
    [
        (0, Description(DummyComponent, str_field="foo", float_field=123.45, list_field=['a', 'b', 3]), True),
        (0, Description(DummyComponent, float_field=123.45), True),
        (0, Description(DummyComponent, float_field=123.46), False),
        (0, Description(DummyComponent, float_field=NumericRange(123.45, rel_tol=0.005)), True),
        (0, Description(DummyComponent, float_field=NumericRange(123.45, abs_tol=0.1)), True),
        (0, Description(DummyComponent, float_field=NumericRange(123.0, end=124.0)), True),
        (0, Description(DummyComponent, float_field=NumericRange(124.0, end=124.1)), False),
        (0, Description(DummyComponent, float_field=NumericRange(124.0, rel_tol=0.001)), False),
        (0, Description(DummyComponent, str_field=Any), True),
        (0, Description(DummyComponent, str_field="foo", list_field=Any), True),
        (0, Description(DummyComponent, str_field="bar", list_field=Any), False),
    ]
)
def test_description_match_unnested(idx, desc, expected):
    assert desc.matches(DummyComponent.detect()[idx]) == expected


@pytest.mark.parametrize(
    "desc",
    [
        Description(DummySystem, subcomponents=[Description(DummyComponent, str_field="foo"),
                                                Description(DummyComponent, str_field="bar")]),
        Description(DummySystem, subcomponents=[Description(DummyComponent, str_field="baz"),
                                                Description(DummyComponent, str_field="bar")]),
        Description(DummySystem, subcomponents=[Description(DummyComponent, float_field=Any),
                                                Description(DummyComponent, str_field="bar",
                                                            float_field=NumericRange(1330))]),
        Description(DummySystem, subcomponents=Any),
    ]
)
def test_description_json(desc):
    s = dumps(desc)
    new_desc = loads(s)
    assert desc == new_desc
