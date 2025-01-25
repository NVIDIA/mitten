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

from nvmitten.interval import Interval, NumericRange
from nvmitten.json_utils import loads, dumps


@pytest.mark.parametrize(
    "a,b,other,expected",
    [
        (1, 5, Interval(0, 2), True),
        (1, 5, Interval(0, 1), True),
        (1, 5, Interval(6, 10), False),
        (2, 1, Interval(1, 1), True),
        (0, 100, Interval(0, 0), True),
        (0, 100, Interval(-10, -1), False),
    ]
)
def test_interval(a, b, other, expected):
    i = Interval(a, b)
    assert i.start == min(a, b)
    assert i.end == max(a, b)
    assert i.intersects(other) == expected


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (1, 5, {1, 2, 3, 4, 5}),
        (-3, 1, {-3, -2, -1, 0, 1}),
        (12, 9, {9, 10, 11, 12}),
    ]
)
def test_interval_to_set(a, b, expected):
    i = Interval(a, b)
    assert i.to_set() == expected


@pytest.mark.parametrize(
    "L,expected",
    [
        ([3, 1, 2, 6, 10, 9, 8], [Interval(1, 3), Interval(6, 6), Interval(8, 10)]),
        ([-1, -2, 0, -2, -1], [Interval(-2, 0)]),
    ]
)
def test_interval_from_list(L, expected):
    assert Interval.build_interval_list(L) == expected

@pytest.mark.parametrize(
    "nr",
    [
        NumericRange(123.456, end=789),
        NumericRange(123, rel_tol=0.03),
        NumericRange(123, abs_tol=321),
    ]
)
def test_numeric_range_json(nr):
    s = dumps(nr)
    new_nr = loads(s)
    assert nr == new_nr
