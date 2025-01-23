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
import sys

from nvmitten.importer import ScopedImporter, import_from


def test_scoped_importer():
    current_sys_path = sys.path[:]

    custom_path = ["tests/assets/scoped_imports"]
    with ScopedImporter(custom_path):
        assert sys.path == custom_path, "sys.path is not set to the expected custom path"
        import my_fake_module
        c = my_fake_module.CustomClass(2)
        assert c.a_func() == 2, "CustomClass does not behave as expected. Possibly imported incorrectly."
    assert sys.path == current_sys_path, "sys.path was not reset correctly"


def test_nested_scoped_importer():
    current_sys_path = sys.path[:]

    custom_path = ["tests/assets/scoped_imports"]
    with ScopedImporter(custom_path):
        assert sys.path == custom_path, "sys.path is not set to the expected custom path"
        import my_fake_module
        c = my_fake_module.CustomClass(2)
        assert c.a_func() == 2, "CustomClass does not behave as expected. Possibly imported incorrectly."

        nested_custom_path = ["tests/assets/scoped_imports/nested"]
        with ScopedImporter(nested_custom_path):
            assert sys.path == nested_custom_path, "sys.path is not set to expected path in nested scope"
            import nested_module
            f = nested_module.Foo(17)
            assert f.bar() == 17, "Foo does not behave as expected. Possibly imported incorrectly."

        assert sys.path == custom_path, "sys.path is not set to the expected custom path after nested scope exit"
    assert sys.path == current_sys_path, "sys.path was not reset correctly in outer scope"


def test_import_from():
    current_sys_path = sys.path[:]
    custom_path = ["tests/assets/scoped_imports"]
    my_fake_module = import_from(custom_path, "my_fake_module")
    assert sys.path == current_sys_path, "sys.path was not reset correctly"
    c = my_fake_module.CustomClass(2)
    assert c.a_func() == 2, "CustomClass does not behave as expected. Possibly imported incorrectly."
