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
import contextlib
import sys
if sys.version_info.major == 3:
    if sys.version_info.minor >= 11:
        import tomllib
    else:
        try:
            import tomli as tomllib
        except:
            import pip._vendor.tomli as tomllib
else:
    raise ModuleNotFoundError("tomllib")
from unittest.mock import MagicMock, patch

from nvmitten.configurator import Field, Configuration, ConfigurationIndex, bind, autoconfigure
from nvmitten.configurator.configurable import *
from nvmitten.importer import import_from

from tests.assets.configurator import my_fields as test_fields


@patch("nvmitten.configurator.configurable.parse_fields")
def test_bind_and_configure(mock_parsefields):
    f1 = Field("f1")

    @bind(f1, "foo")
    class Foo:
        def __init__(self, foo: str = None):
            self.foo = foo

    # Check for defaults
    inst1 = Foo()
    assert inst1.foo is None

    # Check for normal Python behavior
    inst2 = Foo(foo="hello friend")
    assert inst2.foo == "hello friend"

    # Check for parsed Field behavior (empty config)
    mock_parsefields.return_value = {"f1": "abc123"}
    inst3 = Foo.from_fields()
    assert inst3.foo == "abc123"


@patch("nvmitten.configurator.configurable.parse_fields")
def test_from_fields_bound_subset(mock_parsefields):
    f1 = Field("f1")

    @bind(f1, "foo")
    class Foo:
        def __init__(self, foo: str = None, bar: int = 2):
            self.foo = foo
            self.bar = bar

    mock_parsefields.return_value = {"f1": "cool string"}
    inst = Foo.from_fields(bar=1337)
    assert inst.foo == "cool string"
    assert inst.bar == 1337


def test_from_fields_config():
    f1 = Field("f1")
    f2 = Field("f2", from_string=int)

    @bind(f1, "foo")
    @bind(f2, "bar")
    class Foo:
        def __init__(self, foo: str = None, bar: int = 2):
            self.foo = foo
            self.bar = bar


    conf = Configuration({f1: "omg a string",
                          f2: 1234})

    inst = Foo.from_fields(config=conf)
    assert inst.foo == "omg a string"
    assert inst.bar == 1234


def test_from_fields_config_partial():
    f1 = Field("f1")
    f2 = Field("f2", from_string=int)

    @bind(f1, "foo")
    @bind(f2, "bar")
    class Foo:
        def __init__(self, foo: str = None, bar: int = 2):
            self.foo = foo
            self.bar = bar


    conf = Configuration({f2: 1234})

    inst = Foo.from_fields(config=conf)
    assert inst.foo is None
    assert inst.bar == 1234


@patch("nvmitten.configurator.configurable.parse_fields")
def test_from_fields_config_disallow_default(mock_parsefields):
    f1 = Field("f1", disallow_default=True)
    f2 = Field("f2", from_string=int)

    @bind(f1, "foo")
    @bind(f2, "bar")
    class Foo:
        def __init__(self, foo: str = None, bar: int = 2):
            self.foo = foo
            self.bar = bar


    mock_parsefields.return_value = dict()
    with pytest.raises(ValueError):
        Foo.from_fields()


@patch("nvmitten.configurator.configurable.parse_fields")
def test_autoconf(mock_parsefields):
    f1 = Field("f1", disallow_default=True)
    f2 = Field("f2", from_string=int)


    class Bar:
        def __init__(self, baz: bool = False):
            self.baz = baz


    @autoconfigure
    @bind(f1, "foo")
    @bind(f2, "bar")
    class Foo(Bar):
        def __init__(self,
                     foo: str = None,
                     bar: int = 2,
                     baz: bool = False):
            super().__init__(baz=baz)
            self.foo = foo
            self.bar = bar

    assert Foo._mitten_bound_fields[f1] == "foo"
    assert Foo._mitten_bound_fields[f2] == "bar"

    mock_parsefields.return_value = {"f1": "zzz"}
    inst1 = Foo()
    assert inst1.foo == None
    assert inst1.bar == 2
    assert inst1.baz == False

    inst2 = Foo.from_fields()
    assert inst2.foo == "zzz"
    assert inst2.bar == 2
    assert inst2.baz == False

    mock_parsefields.return_value = {"f1": "yyy"}
    with Configuration({f1: "www", f2: 9001}).autoapply():
        inst3 = Foo(baz=True)
    assert inst3.foo == "yyy"
    assert inst3.bar == 9001
    assert inst3.baz == True


def test_configuration_index():
    index = ConfigurationIndex()

    prefix = ["my_system", "FakeGPT"]
    index.load_module("tests.assets.configurator.config", prefix=prefix)


    conf1 = index.get(["default"], prefix=prefix)
    assert [k.name for k in conf1.keys()] == ["batch_size", "use_some_kernel"]
    assert list(conf1.values()) == [12, True]

    conf2 = index.get(["MaxP"], prefix=prefix)
    assert conf2[test_fields.batch_size] == 16
    assert len(conf2) == 1

    conf3 = index.get(["MaxQ"], prefix=prefix)
    assert conf3[test_fields.batch_size] == 8
    assert conf3[test_fields.use_some_kernel] == False
    assert len(conf3) == 2


def test_autoconfigure_with_inherited_dataclasses():
    """Test that autoconfigure works with inherited dataclasses, including parent fields."""
    import dataclasses

    # Create fields for parent and child classes
    parent_field = Field("parent_value")
    child_field = Field("child_value", from_string=int)

    # Define parent dataclass with autoconfigure
    @autoconfigure
    @bind(parent_field, "parent_param")
    @dataclasses.dataclass
    class ParentDataclass:
        parent_param: str = "default_parent"
        other_param: str = "unchanged"

    # Define child dataclass that inherits from parent
    @autoconfigure
    @bind(child_field, "child_param")
    @dataclasses.dataclass
    class ChildDataclass(ParentDataclass):
        child_param: int = 42
        child_only_param: str = "child_default"

    # Create configuration with values for both parent and child fields
    config = Configuration({
        parent_field: "configured_parent_value",
        child_field: 100
    })

    # Test that child class receives both parent and child field configurations
    with config.autoapply():
        instance = ChildDataclass()

    # Verify both parent and child fields are configured
    assert instance.parent_param == "configured_parent_value"
    assert instance.child_param == 100
    assert instance.other_param == "unchanged"  # Non-bound field remains default
    assert instance.child_only_param == "child_default"  # Non-bound field remains default


def test_autoconfigure_inherited_dataclass_partial_config():
    """Test autoconfigure with inherited dataclasses when only some fields are configured."""
    import dataclasses

    # Create fields
    parent_field1 = Field("parent_val1")
    parent_field2 = Field("parent_val2")
    child_field = Field("child_val", from_string=int)

    # Parent dataclass with multiple bound fields
    @autoconfigure
    @bind(parent_field1, "param1")
    @bind(parent_field2, "param2")
    @dataclasses.dataclass
    class ParentClass:
        param1: str = "default1"
        param2: str = "default2"

    # Child dataclass
    @autoconfigure
    @bind(child_field, "param3")
    @dataclasses.dataclass
    class ChildClass(ParentClass):
        param3: int = 0

    # Configuration with only some fields set
    config = Configuration({
        parent_field1: "configured1",
        child_field: 999
        # parent_field2 is not configured, should use default
    })

    with config.autoapply():
        instance = ChildClass()

    # Verify configured fields are set, others use defaults
    assert instance.param1 == "configured1"
    assert instance.param2 == "default2"  # Uses default since not in config
    assert instance.param3 == 999


def test_autoconfigure_multiple_inheritance_levels():
    """Test autoconfigure with multiple levels of dataclass inheritance."""
    import dataclasses

    # Create fields for each level
    grandparent_field = Field("gp_value")
    parent_field = Field("p_value")
    child_field = Field("c_value")

    # Grandparent dataclass
    @autoconfigure
    @bind(grandparent_field, "gp_param")
    @dataclasses.dataclass
    class GrandparentClass:
        gp_param: str = "gp_default"

    # Parent dataclass
    @autoconfigure
    @bind(parent_field, "p_param")
    @dataclasses.dataclass
    class ParentClass(GrandparentClass):
        p_param: str = "p_default"

    # Child dataclass
    @autoconfigure
    @bind(child_field, "c_param")
    @dataclasses.dataclass
    class ChildClass(ParentClass):
        c_param: str = "c_default"

    # Configuration with all fields
    config = Configuration({
        grandparent_field: "gp_configured",
        parent_field: "p_configured",
        child_field: "c_configured"
    })

    with config.autoapply():
        instance = ChildClass()

    # Verify all fields from all inheritance levels are configured
    assert instance.gp_param == "gp_configured"
    assert instance.p_param == "p_configured"
    assert instance.c_param == "c_configured"