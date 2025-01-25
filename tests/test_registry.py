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

from nvmitten.registry import *


class Example:
    pass


class TestRegistry:
    """Tests Registry API
    """

    def test_register(self):
        ExampleRegistry = Registry(Example)
        example1 = Example()
        example2 = Example()
        example3 = Example()

        ExampleRegistry.register(("examples", "1"), example1)
        ExampleRegistry.register(("examples", "2"), example2)
        ExampleRegistry.register(("examples", "3"), example3)

        assert ExampleRegistry.get(("examples", "1")) is example1
        assert ExampleRegistry.get(("examples", "2")) is example2
        assert ExampleRegistry.get(("examples", "3")) is example3
        assert type(ExampleRegistry.get("examples")) is list
        assert len(ExampleRegistry.get("examples")) == 3

    def test_register_class(self):
        ExampleRegistry = Registry(Example, register_instances=False)

        @ExampleRegistry.register(("examples", "1"))
        class Example1(Example):
            pass

        @ExampleRegistry.register(("examples", "2"))
        class Example2(Example):
            pass

        @ExampleRegistry.register(("examples", "3"))
        class Example3(Example):
            pass

        assert ExampleRegistry.get(("examples", "1")) is Example1
        assert ExampleRegistry.get(("examples", "2")) is Example2
        assert ExampleRegistry.get(("examples", "3")) is Example3
        assert type(ExampleRegistry.get("examples")) is list
        assert len(ExampleRegistry.get("examples")) == 3
