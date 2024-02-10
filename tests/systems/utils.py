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

import os

from nvmitten.systems.info_source import INFO_SOURCE_REGISTRY


def force_reset(info_sources):
    for src in info_sources:
        src.reset(hard=True)


def spoof_wrapper(spoof_node_id, info_source_keys, f, visible_devices=None, setup=None, cleanup=None):
    # Override $PATH
    spoof_script_dir = os.path.join(os.getcwd(), "tests", "assets", "system_detect_spoofs")
    old_path = os.environ["PATH"]
    new_path = ":".join([spoof_script_dir, old_path])
    old_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)

    try:
        os.environ["PATH"] = new_path
        os.environ["_SPOOF_NODE_ID"] = spoof_node_id
        if visible_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices

        if setup is not None:
            setup()

        force_reset(map(INFO_SOURCE_REGISTRY.get, info_source_keys))

        f()
    finally:
        os.environ["PATH"] = old_path
        os.environ["_SPOOF_NODE_ID"] = ""
        if visible_devices is not None:
            if old_visible_devices is None:
                del os.environ["CUDA_VISIBLE_DEVICES"]
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = old_visible_devices

        if cleanup is not None:
            cleanup()

        force_reset(map(INFO_SOURCE_REGISTRY.get, info_source_keys))
