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
import random
import re

from nvmitten.aliased_name import AliasedName, AliasedNameEnum


def randomize_cases(s: str):
    """
    Returns a string 't' such that s and t are case-insensitive equivalent, but 't' has randomized capitalization of its
    characters from 's'.

    Args:
        s (str):
            String to randomize

    Returns:
        str: A string with randomized capitalizations such that it is case-insensitive equivalent to the input string.
    """
    return ''.join(
        random.choice((str.upper, str.lower))(c)
        for c in s
    )


@pytest.mark.parametrize(
    "name,aliases,other_cases",
    [
        ("", ("naDa", "nothing", "empty"), {False: (None, AliasedName(None), "zilch", AliasedName("nada"))}),
        ("foo", ("phoo", "FU", "pHu"), {False: (None, AliasedName(None), "fou", AliasedName("baz"))}),
        (None, tuple(), {False: ("fou", AliasedName("baz")), True: (None, AliasedName(None))}),
    ]
)
def test_aliased_name_eq(name, aliases, other_cases):
    aliased_name = AliasedName(name, aliases)

    for s in [name, *aliases]:
        assert aliased_name == s
        if type(s) is str:
            assert aliased_name == randomize_cases(s)
            assert aliased_name == s.lower()
            assert aliased_name == s.upper()

    for expected_value, cases in other_cases.items():
        for case in cases:
            assert expected_value == (aliased_name == case)


@pytest.mark.parametrize(
    "name,aliases",
    [
        (None, tuple()),
        ("foo", ("phoo", "fu", "phu")),
        ("BAR", ("baz", "bbbar")),
        ("", ("empty", "nothing")),
    ]
)
def test_aliased_name_hash(name, aliases):
    aliased_name = AliasedName(name, aliases)
    expected = hash(None if name is None else name.lower())
    assert expected == hash(aliased_name)
    if type(name) is str:
        assert expected == hash(AliasedName(randomize_cases(name), aliases))
    for s in aliases:
        if s != name:
            assert hash(s) != hash(aliased_name)
            assert hash(s.lower()) != hash(aliased_name)


def test_aliased_name_regex():
    aliased_name = AliasedName("foo", aliases=("fou", "fu"), patterns=(re.compile(r"foo+"), re.compile(r"ph[ou]")))
    assert aliased_name == randomize_cases("fou")
    assert aliased_name == randomize_cases("fu")
    assert aliased_name == randomize_cases("foo")
    assert aliased_name == "fooo"
    assert aliased_name != "fooO"
    assert aliased_name == "pho"
    assert aliased_name != "phoo"
    assert aliased_name != "phou"
    assert aliased_name != 2
    assert aliased_name != None
    assert aliased_name != "phOu"


def test_aliased_name_enum():
    class DummyEnum(AliasedNameEnum):
        Foo: AliasedName = AliasedName("foo", ("fu", "phou"))
        Bar: AliasedName = AliasedName("BaR", ("baar", "bbbAR"))

    assert len(DummyEnum) == 2
    assert DummyEnum.get_match(randomize_cases("bar")) is DummyEnum.Bar
    assert DummyEnum.get_match("baz") is None
    assert DummyEnum.Foo == "FU"  # __eq__ forwarding
