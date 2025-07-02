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

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from nvmitten.pipeline.resource import (
    SourceURL,
    LocalDataSource,
    GETableSource,
    Checksum,
    MD5Checksum,
    Resource
)
from nvmitten.pipeline.errors import (
    ResourceNotFoundError,
    ResourceAccessError,
    ResourceDownloadError,
    ResourceValidationError,
    ChecksumError
)
from nvmitten.pipeline.scratch_space import ScratchSpace


class TestSourceURL:
    def test_abstract_method(self):
        with pytest.raises(NotImplementedError):
            SourceURL().retrieve_to(Path("test"))


class TestLocalDataSource:
    def test_init(self):
        data_source = LocalDataSource("test/path")
        assert isinstance(data_source.fpath, Path)
        assert str(data_source.fpath) == "test/path"

    def test_retrieve_to_file(self, tmp_path):
        # Create source file
        src_file = tmp_path / "source.txt"
        src_file.write_text("test content")
        
        # Create destination path
        dst_file = tmp_path / "dest.txt"
        
        # Test file copy
        data_source = LocalDataSource(src_file)
        data_source.retrieve_to(dst_file)
        
        assert dst_file.exists()
        assert dst_file.read_text() == "test content"

    def test_retrieve_to_directory(self, tmp_path):
        # Create source directory with files
        src_dir = tmp_path / "source"
        src_dir.mkdir()
        (src_dir / "file1.txt").write_text("file1")
        (src_dir / "file2.txt").write_text("file2")
        
        # Create destination path
        dst_dir = tmp_path / "dest"
        
        # Test directory copy
        data_source = LocalDataSource(src_dir)
        data_source.retrieve_to(dst_dir)
        
        assert dst_dir.exists()
        assert (dst_dir / "file1.txt").read_text() == "file1"
        assert (dst_dir / "file2.txt").read_text() == "file2"

    def test_retrieve_to_nonexistent_source(self, tmp_path):
        data_source = LocalDataSource(tmp_path / "nonexistent")
        with pytest.raises(ResourceNotFoundError):
            data_source.retrieve_to(tmp_path / "dest")

    def test_retrieve_to_invalid_source(self, tmp_path):
        # Create a special file (not regular file or directory)
        special_file = tmp_path / "special"
        os.mkfifo(str(special_file))
        
        data_source = LocalDataSource(special_file)
        with pytest.raises(ResourceValidationError):
            data_source.retrieve_to(tmp_path / "dest")


class TestGETableSource:
    def test_init(self):
        source = GETableSource("http://test.com")
        assert source.url == "http://test.com"
        assert source.allow_redirects is True
        assert source.chunk_size == 4 * 1024
        assert source.timeout == 30

    @patch('requests.get')
    def test_retrieve_to_success(self, mock_get, tmp_path):
        # Mock response
        mock_response = MagicMock()
        mock_response.headers = {'content-length': '100'}
        mock_response.iter_content.return_value = [b'chunk1', b'chunk2']
        mock_get.return_value = mock_response
        
        # Test download
        dst_file = tmp_path / "test.txt"
        source = GETableSource("http://test.com")
        source.retrieve_to(dst_file)
        
        assert dst_file.exists()
        mock_get.assert_called_once()
        mock_response.iter_content.assert_called_once()

    @patch('requests.get')
    def test_retrieve_to_no_content_length(self, mock_get):
        mock_response = MagicMock()
        mock_response.headers = {}
        mock_get.return_value = mock_response
        
        source = GETableSource("http://test.com")
        with pytest.raises(ResourceDownloadError):
            source.retrieve_to(Path("test.txt"))


class TestChecksum:
    def test_abstract_method(self):
        with pytest.raises(NotImplementedError):
            Checksum("test").evaluate(b"data")

    def test_verify_success(self, tmp_path):
        class TestChecksumImpl(Checksum):
            def evaluate(self, bytestr):
                return "test"
        
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        
        checksum = TestChecksumImpl("test")
        checksum.verify(test_file)  # Should not raise

    def test_verify_failure(self, tmp_path):
        class TestChecksumImpl(Checksum):
            def evaluate(self, bytestr):
                return "wrong"
        
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        
        checksum = TestChecksumImpl("test")
        with pytest.raises(ChecksumError):
            checksum.verify(test_file)

    def test_verify_file_error(self):
        checksum = MD5Checksum("test")
        with pytest.raises(ResourceAccessError):
            checksum.verify(Path("nonexistent"))


class TestMD5Checksum:
    def test_evaluate(self):
        checksum = MD5Checksum("test")
        result = checksum.evaluate(b"test content")
        assert isinstance(result, str)
        assert len(result) == 32  # MD5 hash length
        assert result == "9473fdd0d880a43c21b7778d34872157"


class TestResource:
    def test_init(self):
        resource = Resource("test/path")
        assert isinstance(resource.resource_path, Path)
        assert str(resource.resource_path) == "test/path"
        assert resource.source_url is None
        assert resource.subresources is None
        assert resource.checksum is None

    def test_path_traversal_protection(self):
        with pytest.raises(ResourceValidationError):
            Resource("../test/path")

    def test_eff_path(self):
        resource = Resource("test/path")
        assert resource.eff_path() == Path("test/path")
        
        scratch = ScratchSpace(Path("scratch"))
        assert resource.eff_path(scratch) == Path("scratch/test/path")

    def test_exists(self, tmp_path):
        scratch = ScratchSpace(tmp_path)

        # Test with no subresources
        resource = Resource("test.txt")
        assert not resource.exists(scratch_space=scratch)
        
        # Create file
        (tmp_path / "test.txt").touch()
        assert resource.exists(scratch_space=scratch)
        
        # Test with subresources
        subresource = Resource("sub/test.txt")
        resource = Resource("test.txt", subresources=[subresource])
        assert not resource.exists(scratch_space=scratch)  # Subresource doesn't exist
        
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub/test.txt").touch()
        assert resource.exists(scratch_space=scratch)

    def test_destroy(self, tmp_path):
        # Create test files
        main_file = tmp_path / "test.txt"
        main_file.touch()
        sub_dir = tmp_path / "sub"
        sub_dir.mkdir()
        sub_file = sub_dir / "test.txt"
        sub_file.touch()
        
        # Create resource with subresource
        subresource = Resource("sub/test.txt")
        resource = Resource("test.txt", subresources=[subresource])
        
        # Test destroy
        resource.destroy(scratch_space=ScratchSpace(tmp_path))
        assert not main_file.exists()
        assert not sub_file.exists()

    def test_verify(self, tmp_path):
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        
        # Create MD5 checksum
        checksum = MD5Checksum("9473fdd0d880a43c21b7778d34872157")  # MD5 of "test content"
        resource = Resource("test.txt", checksum=checksum)
        
        # Test verify
        resource.verify(scratch_space=ScratchSpace(tmp_path))  # Should not raise
        
        # Test with wrong checksum
        wrong_checksum = MD5Checksum("wrong")
        resource = Resource("test.txt", checksum=wrong_checksum)
        with pytest.raises(ChecksumError):
            resource.verify(scratch_space=ScratchSpace(tmp_path))

    def test_create(self, tmp_path):
        # Test with no source_url
        resource = Resource("test.txt")
        resource.create(scratch_space=ScratchSpace(tmp_path))
        assert not (tmp_path / "test.txt").exists()  # Should not create anything
        
        # Test with source_url
        test_file = tmp_path / "source.txt"
        test_file.write_text("test content")
        source = LocalDataSource(test_file)
        resource = Resource("test.txt", source_url=source)
        
        resource.create(scratch_space=ScratchSpace(tmp_path))
        assert (tmp_path / "test.txt").exists()
        assert (tmp_path / "test.txt").read_text() == "test content"
        
        # Test with force
        test_file.write_text("new content")
        resource.create(scratch_space=ScratchSpace(tmp_path), force=True)
        assert (tmp_path / "test.txt").read_text() == "new content" 