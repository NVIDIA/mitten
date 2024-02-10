# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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


class InvalidImplError(TypeError):
    def __init__(self, op, fake_impl):
        super().__init__(f"Operation {op} implements {fake_impl}, which is not subclass of Impl.")


class TooManyImplementationsError(RuntimeError):
    def __init__(self, impl, ops):
        super().__init__(f"{impl} has too many implementing Operations: {ops}")


class ImplementationNotFoundError(RuntimeError):
    def __init__(self, op, impl):
        super().__init__(f"{op} depends on Impl {impl} but no implementation was given.")


class FailedOperationError(RuntimeError):
    def __init__(self, op, output):
        super().__init__(f"{op} failed with output {output}")
