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

from nvmitten.matchable import *


@pytest.mark.parametrize(
    "matcher,other,expected",
    [
        (MatchAllowList([2, 17, 19, "foo"]), 2, True),
        (MatchAllowList([2, 17, 19, "foo"]), 19, True),
        (MatchAllowList([2, 17, 19, "foo"]), 17, True),
        (MatchAllowList([2, 17, 19, "foo"]), "foo", True),
        (MatchAllowList([2, 17, 19, "foo"]), "2", False),
        (MatchAllowList([2, 17, 19, "foo"]), None, False),
        (MatchAllowList([2, 17, 19, "foo"]), [2, 17], False),
        (MatchAllowList([None]), None, True),
        (MatchAllowList([]), MATCH_ANY, True),
        (MatchAllowList([]), 17, False),
    ]
)
def test_matchallowlist(matcher, other, expected):
    assert (matcher == other) == expected


@pytest.mark.parametrize(
    "other",
    [
        0,
        None,
        17.49,
        "hi",
        MatchAllowList(["bar"]),
        True,
        False,
        {"a": 1, 2: "b"},
        [1, 2, 3],
        range(27),
        int,
        pytest,
    ]
)
def test_matchany(other):
    # MATCH_ANY always returns True
    assert MATCH_ANY == other


@pytest.mark.parametrize(
    "matcher,other,expected",
    [
        (MatchFloatApproximate(28., float), 27.9991, True),
        (MatchFloatApproximate(28., float), 29., True),
        (MatchFloatApproximate(28., float), 30., False),
    ]
)
def test_matchfloatapproximate(matcher, other, expected):
    assert (matcher == other) == expected


@pytest.mark.parametrize(
    "matcher,other,expected",
    [
        (MatchNumericThreshold(10, int), 10, True),
        (MatchNumericThreshold(10, int), 11, True),
        (MatchNumericThreshold(10, int), 100, True),
        (MatchNumericThreshold(10, int), 9, False),
        (MatchNumericThreshold(10, int), "10", False),
        (MatchNumericThreshold("foobarbaz", len, min_threshold=False), "a string", True),
        (MatchNumericThreshold("foobarbaz", len, min_threshold=False), "a quick brown fox", False),
    ]
)
def test_matchnumericthreshold(matcher, other, expected):
    assert (matcher == other) == expected
