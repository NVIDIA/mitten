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


class MissingParentOutputKey(KeyError):
    def __init__(self, cls, parent, keys):
        msg = f"{cls} is missing output keys {keys}, which is required by parent {parent}"
        super().__init__(msg)


class ImplementationNotFoundError(RuntimeError):
    def __init__(self, op, impl):
        super().__init__(f"{op} depends on {impl} but no implementation was found.")


class ResourceError(RuntimeError):
    """Base class for resource-related errors."""
    def __init__(self, message: str):
        super().__init__(message)


class ChecksumError(ResourceError):
    """Raised when checksum verification fails."""
    def __init__(self, message: str):
        super().__init__(message)


class ResourceNotFoundError(ResourceError):
    """Raised when a requested resource cannot be found."""
    def __init__(self, path: str):
        super().__init__(f"Resource not found: {path}")


class ResourceAccessError(ResourceError):
    """Raised when there are permission issues accessing a resource."""
    def __init__(self, path: str, operation: str):
        super().__init__(f"Failed to {operation} resource at {path}")


class ResourceDownloadError(ResourceError):
    """Raised when downloading a resource fails."""
    def __init__(self, url: str, reason: str):
        super().__init__(f"Failed to download resource from {url}: {reason}")


class ResourceValidationError(ResourceError):
    """Raised when a resource fails validation checks."""
    def __init__(self, path: str, reason: str):
        super().__init__(f"Resource validation failed for {path}: {reason}")
