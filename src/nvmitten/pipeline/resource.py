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

"""Resource management module for handling file and directory resources.

This module provides classes for managing resources in a pipeline, including:
- SourceURL: Abstract base class for resource sources
- LocalDataSource: Local file/directory source implementation
- GETableSource: HTTP source implementation
- Checksum: Abstract base class for checksum verification
- MD5Checksum: MD5 checksum implementation
- Resource: Main class for managing resources with source, subresources, and checksum
"""

from __future__ import annotations
from pathlib import Path
from requests.exceptions import HTTPError, RequestException
from tqdm import tqdm
from typing import List, Optional

import collections
import dataclasses as dcls
import hashlib
import logging
import os
import requests
import shutil
import tempfile

from .scratch_space import ScratchSpace
from .errors import (
    ResourceError,
    ChecksumError,
    ResourceNotFoundError,
    ResourceAccessError,
    ResourceDownloadError,
    ResourceValidationError
)


class SourceURL:
    """Abstract base class for resource sources.

    This class defines the interface for retrieving resources from various sources.
    Implementations must provide a retrieve_to method to copy the resource to a destination.
    """

    def retrieve_to(self, dst: Path):
        """Retrieve the resource to the specified destination path.

        Args:
            dst: Destination path where the resource should be copied to.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError


class LocalDataSource(SourceURL):
    """Local file or directory data source implementation.

    This class handles copying resources from the local filesystem.
    """

    def __init__(self, fpath: os.PathLike):
        """Initialize a local data source.

        Args:
            fpath: Path to the local file or directory.
        """
        self.fpath = Path(fpath)

    def retrieve_to(self, dst: Path):
        """Copy the local resource to the destination path.

        Args:
            dst: Destination path where the resource should be copied to.

        Raises:
            ResourceNotFoundError: If the source file/directory does not exist.
            ResourceValidationError: If the source is not a file or directory.
            ResourceAccessError: If there is an error during the copy operation.
        """
        if not self.fpath.exists():
            raise ResourceNotFoundError(str(self.fpath))
        try:
            if self.fpath.is_file():
                shutil.copy(self.fpath, dst)
            elif self.fpath.is_dir():
                shutil.copytree(self.fpath, dst)
            else:
                raise ResourceValidationError(str(self.fpath), "not a file or directory")
        except (OSError, IOError) as e:
            raise ResourceAccessError(str(self.fpath), f"copy: {str(e)}")


class GETableSource(SourceURL):
    """HTTP source implementation for downloading resources from URLs.

    This class handles downloading resources from HTTP/HTTPS URLs with progress tracking
    and atomic writes.
    """

    def __init__(self,
                 url: str,
                 allow_redirects: bool = True,
                 chunk_size: int = 4 * 1024,
                 timeout: int = 30):
        """Initialize a GETable source.

        Args:
            url: The URL to download from.
            allow_redirects: Whether to follow HTTP redirects.
            chunk_size: Size of chunks to download at a time.
            timeout: Request timeout in seconds.
        """
        self.url = url
        self.allow_redirects = allow_redirects
        self.chunk_size = chunk_size
        self.timeout = timeout

    def retrieve_to(self, dst: Path):
        """Download the resource from the URL to the destination path.

        Args:
            dst: Destination path where the resource should be downloaded to.

        Raises:
            ResourceDownloadError: If there is an error during download, including:
                - Missing content-length header
                - Network errors
                - Write errors
        """
        try:
            resp = requests.get(
                self.url,
                stream=True,
                allow_redirects=self.allow_redirects,
                timeout=self.timeout
            )
            resp.raise_for_status()

            if "content-length" not in resp.headers:
                raise ResourceDownloadError(self.url, "could not retrieve file-size information")
            file_size = int(resp.headers["content-length"])

            # Create a temporary file for atomic write
            with tempfile.NamedTemporaryFile(delete=False, dir=dst.parent) as temp_file:
                temp_path = Path(temp_file.name)
                tqdm_desc = f"{self.url[:12]} -> {str(dst)[:12]}"
                pbar = tqdm(total=file_size,
                          desc=tqdm_desc,
                          leave=True)
                try:
                    for data in resp.iter_content(chunk_size=self.chunk_size):
                        temp_file.write(data)
                        pbar.update(len(data))
                    pbar.close()
                    temp_file.flush()
                    os.fsync(temp_file.fileno())

                    # Atomic move to final destination
                    temp_path.replace(dst)
                except Exception as e:
                    # Clean up temp file on error
                    temp_path.unlink(missing_ok=True)
                    raise ResourceDownloadError(self.url, str(e))
        except RequestException as e:
            raise ResourceDownloadError(self.url, f"network error: {str(e)}")


class Checksum:
    """Abstract base class for checksum verification.

    This class defines the interface for verifying resource integrity using checksums.
    """

    def __init__(self, s):
        """Initialize a checksum verifier.

        Args:
            s: The expected checksum value.
        """
        self.s = s

    def evaluate(self, bytestr) -> str:
        """Calculate the checksum of the given bytes.

        Args:
            bytestr: The bytes to calculate the checksum for.

        Returns:
            The calculated checksum as a string.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def verify(self, dst: Path):
        """Verify the checksum of the file at the given path.

        Args:
            dst: Path to the file to verify.

        Raises:
            ChecksumError: If the checksum verification fails.
            ResourceAccessError: If there is an error reading the file.
        """
        try:
            with Path(dst).open(mode="rb") as handle:
                actual = self.evaluate(handle.read()).lower()
                expected = self.s.lower()
                if actual != expected:
                    raise ChecksumError(
                        f"Checksum verification failed for {dst}. "
                        f"Expected: {expected}, Got: {actual}"
                    )
        except (IOError, OSError) as e:
            raise ResourceAccessError(str(dst), f"verify checksum: {str(e)}")


class MD5Checksum(Checksum):
    """MD5 checksum implementation.

    This class provides MD5 hash calculation for resource verification.
    """

    def evaluate(self, bytestr) -> str:
        """Calculate the MD5 hash of the given bytes.

        Args:
            bytestr: The bytes to calculate the MD5 hash for.

        Returns:
            The MD5 hash as a hexadecimal string.
        """
        return hashlib.md5(bytestr).hexdigest()


@dcls.dataclass
class Resource:
    """Main class for managing resources in a pipeline.

    A resource represents a file or directory that can be:
    - Retrieved from a source (local or remote)
    - Verified using a checksum
    - Composed of subresources
    - Managed in a scratch space

    Attributes:
        resource_path: Path to the resource relative to the scratch space.
        source_url: Optional source to retrieve the resource from.
        subresources: Optional list of subresources that make up this resource.
        checksum: Optional checksum to verify the resource integrity.
    """

    resource_path: os.PathLike
    source_url: Optional[SourceURL] = None
    subresources: Optional[List[Resource]] = None
    checksum: Optional[Checksum] = None

    def __post_init__(self):
        """Initialize the resource and validate its path.

        Raises:
            ResourceValidationError: If the path contains parent directory traversal.
        """
        self.resource_path = Path(self.resource_path)
        # Validate path components
        if ".." in self.resource_path.parts:
            raise ResourceValidationError(str(self.resource_path), "Path traversal to parent directories is not allowed")

    def eff_path(self, scratch_space: Optional[ScratchSpace] = None) -> Path:
        """Get the effective path of the resource within a scratch space.

        Args:
            scratch_space: Optional scratch space to resolve the path against.

        Returns:
            The absolute path to the resource.
        """
        if scratch_space:
            return scratch_space.path / self.resource_path
        else:
            return self.resource_path

    def exists(self, scratch_space: Optional[ScratchSpace] = None) -> bool:
        """Check if the resource exists.

        Args:
            scratch_space: Optional scratch space to check in.

        Returns:
            True if the resource and all its subresources exist, False otherwise.
        """
        e = self.eff_path(scratch_space).exists()
        if self.subresources:
            for r in self.subresources:
                e = e and r.exists(scratch_space)
        return e

    def destroy(self, scratch_space: Optional[ScratchSpace] = None):
        """Destroy the resource and its subresources.

        Args:
            scratch_space: Optional scratch space containing the resource.

        Raises:
            ResourceAccessError: If there is an error destroying the resource.
        """
        if self.subresources:
            for r in self.subresources:
                try:
                    r.destroy(scratch_space)
                except Exception as e:
                    logging.warning(f"Failed to destroy subresource {r.resource_path}: {str(e)}")

        eff_path = self.eff_path(scratch_space)
        try:
            if eff_path.is_file():
                eff_path.unlink()
            elif eff_path.is_dir():
                shutil.rmtree(eff_path)
        except Exception as e:
            logging.error(f"Failed to destroy resource {eff_path}: {str(e)}")
            raise ResourceAccessError(str(eff_path), f"destroy: {str(e)}")

    def verify(self, scratch_space: Optional[ScratchSpace] = None):
        """Verify the resource and its subresources.

        Args:
            scratch_space: Optional scratch space containing the resource.

        Raises:
            ChecksumError: If checksum verification fails.
            ResourceAccessError: If there is an error accessing the resource.
        """
        if self.subresources:
            for r in self.subresources:
                r.verify(scratch_space)

        if self.checksum:
            eff_path = self.eff_path(scratch_space)
            self.checksum.verify(eff_path)

    def create(self,
               scratch_space: Optional[ScratchSpace] = None,
               force: bool = False):
        """Create the resource from its source.

        Args:
            scratch_space: Optional scratch space to create the resource in.
            force: Whether to force recreation if the resource already exists.

        Returns:
            True if the resource was created or already exists, False otherwise.

        Raises:
            ResourceAccessError: If there is an error creating the resource.
        """
        if not force and self.exists(scratch_space):
            return True
        elif force:
            self.destroy(scratch_space)

        if self.source_url:
            eff_path = self.eff_path(scratch_space)
            try:
                # Ensure parent directory exists
                eff_path.parent.mkdir(parents=True, exist_ok=True)
                self.source_url.retrieve_to(eff_path)
            except (OSError, IOError) as e:
                raise ResourceAccessError(str(eff_path), f"create directory: {str(e)}")

        self.verify(scratch_space)
