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
from dataclasses import dataclass, field
from enum import Enum, unique, auto
from pathlib import Path
from requests.exceptions import HTTPError
from tqdm import tqdm
from typing import Union

import collections
import contextlib
import hashlib
import logging
import requests
import shutil


@unique
class RemoteObjectType(Enum):
    LocalCopy = auto()
    GETable = auto()
    # TODO: Implement object retrieval for the following types
    # GDrive = auto()
    # FTP = auto()
    # Git = auto()
    # ...


@unique
class ChecksumType(Enum):
    MD5 = auto()
    # TODO: Implement other common object hash checksums
    # SHA1 = auto()


@dataclass
class Resource:
    """Represents a file resource.
    """
    resource_path: Union[str, Path]
    """Path: The local relative path this resource is located within the scratch space."""

    scratch_space: ScratchSpace
    """ScratchSpace: The scratch space this resource should be retrieved to."""

    resource_remote_url: str = None
    """str: A URL indicating where the resource should be downloaded from."""

    resource_remote_type: RemoteObjectType = RemoteObjectType.GETable
    """RemoteObjectType: The protocol the remote object should be downloaded with.
    (Default: RemoteObjectType.GETable)"""

    resource_checksum: str = None
    """str: The checksum of the remote object. Used to verify successful downloads."""

    resource_checksum_type: ChecksumType = ChecksumType.MD5
    """ChecksumType: The algorithm used to generate the checksum string. (Default: ChecksumType.MD5)"""

    resource_format: str = "bin"
    """str: The file format of the resource. Used to determine how to unpack the object. (Default: 'bin')"""

    resource_extract_dir: Union[str, Path] = None
    """Path: Path to the directory to unpack the contents of this resource to."""

    require_keep_archive: bool = False
    """bool: If True and this resource is an extractable resource, the original unpacked archive is required to be kept
    for the resource to be considered 'ready for use'. (Default: False)"""

    is_archive: bool = field(init=False)

    def __post_init__(self):
        # Enforce that paths are instances of pathlib.Path
        if not isinstance(self.resource_path, Path):
            self.resource_path = Path(self.resource_path)
        if self.resource_extract_dir and not isinstance(self.resource_extract_dir, Path):
            self.resource_extract_dir = Path(self.resource_extract_dir)

        # Clear the resource_extract_dir if resource_format is not a recognized archive format.
        self.is_archive = self.resource_format in (t[0] for t in shutil.get_unpack_formats())
        if not self.is_archive:
            self.resource_extract_dir = None

    def effective_resource_path(self):
        return self.scratch_space.path / self.resource_path

    def effective_extract_dir(self):
        if self.resource_extract_dir:
            return self.scratch_space.path / self.resource_extract_dir
        else:
            return None

    def ready(self) -> bool:
        eff_extract_dir = self.effective_extract_dir()
        if eff_extract_dir:
            if not eff_extract_dir.is_dir():
                return False

            # TODO: Implement a more robust check. This might need to be done by the implementations themselves.
            if not any(eff_extract_dir.glob('*')):
                return False

        if (eff_extract_dir is None) or self.require_keep_archive:
            if not self.effective_resource_path().exists():
                return False

            if not self.verify_checksum():
                return False
        return True

    def download(self):
        """Downloads the resource.

        Raises:
            requests.exceptions.HTTPError: If the download failed.
        """
        eff_resource_path = self.effective_resource_path()
        eff_resource_path.parent.mkdir(parents=True, exist_ok=True)

        if self.resource_remote_type is RemoteObjectType.LocalCopy:
            src = Path(self.resource_remote_url)
            if not src.exists():
                raise FileNotFoundError(f"Cannot copy non-existent resource {src}")

            if src.is_file():
                shutil.copy(src, eff_resource_path)
            elif src.is_dir():
                shutil.copytree(src, eff_resource_path)
        elif self.resource_remote_type is RemoteObjectType.GETable:
            resp = requests.get(self.resource_remote_url, stream=True, allow_redirects=True)
            resp.raise_for_status()

            file_size = int(resp.headers.get("content-length", -1))
            if file_size == -1:
                raise HTTPError(f"Could not get file size from {self.resource_remote_url}")

            tqdm_desc = f"{self.resource_remote_url[:12]} -> {self.resource_path[:12]}"
            pbar = tqdm(total=file_size,
                        desc=tqdm_desc,
                        leave=True)
            with open(eff_resource_path, 'wb') as handle:
                chunk_size = self._chunk_size if hasattr(self, "_chunk_size") else 4 * 1024
                for data in resp.iter_content(chunk_size=chunk_size):
                    handle.write(data)
                    pbar.update(len(data))
            pbar.close()

    def verify_checksum(self) -> bool:
        """Verifies the checksum of the downloaded resource. If no checksum is provided, this method returns True by
        default.

        Returns:
            bool: Whether or not the checksum matches.
        """
        if self.resource_checksum is None:
            return True

        if self.resource_checksum_type is ChecksumType.MD5:
            with open(self.effective_resource_path(), 'rb') as handle:
                md5_checksum = hashlib.md5(handle.read()).hexdigest()
                if md5_checksum != self.resource_checksum:
                    logging.error(f"{self.resource_checksum_type.name} checksum mismatch: Got '{md5_checksum}', "
                                  f"Expected '{self.resource_checksum}'")
                    return False
        return True

    def extract(self):
        """If this resource is an archive, unpacks it to `self.effective_extract_dir()`.
        """
        eff_extract_dir = self.effective_extract_dir()
        if self.is_archive and eff_extract_dir:
            eff_extract_dir.mkdir(parents=True, exist_ok=True)
            shutil.unpack_archive(self.effective_resource_path(), eff_extract_dir, self.resource_format)

    def generate(self, force: bool = False):
        """If this resource is not ready, downloads and unpacks this archive, when applicable.

        Args:
            force (bool): If True, will generate the resource even if it already exists.
        """
        eff_resource_path = self.effective_resource_path()
        eff_extract_dir = self.effective_extract_dir()

        # Check if we need to download
        if (eff_extract_dir is None) or self.require_keep_archive:
            requires_download = (not eff_resource_path.exists()) or \
                                (not self.verify_checksum()) or \
                                force
            if requires_download:
                if eff_resource_path.is_file():
                    eff_resource_path.unlink()
                elif eff_resource_path.is_dir():
                    shutil.rmtree(eff_resource_path, ignore_errors=True)
                self.download()

        # Check if we need to unpack
        if eff_extract_dir:
            requires_extract = (not eff_extract_dir.is_dir()) or \
                               (not any(eff_extract_dir.glob('*'))) or \
                               force
            if requires_extract:
                if eff_extract_dir.is_file():
                    eff_extract_dir.unlink()
                elif eff_extract_dir.is_dir():
                    shutil.rmtree(eff_extract_dir, ignore_errors=True)
                self.extract()
