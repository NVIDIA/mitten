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

from typing import Any, Iterable, Dict

from ...pipeline import Operation

import mlperf_loadgen as lg
import logging

# Define a map to convert test mode input string into its corresponding enum value
test_mode_map = {
    "SubmissionRun": lg.TestMode.SubmissionRun,
    "AccuracyOnly": lg.TestMode.AccuracyOnly,
    "PerformanceOnly": lg.TestMode.PerformanceOnly,
    "FindPeakPerformance": lg.TestMode.FindPeakPerformance,
}

# Define a map to convert logging mode input string into its corresponding enum value
log_mode_map = {
    "AsyncPoll": lg.LoggingMode.AsyncPoll,
    "EndOfTestOnly": lg.LoggingMode.EndOfTestOnly,
    "Synchronous": lg.LoggingMode.Synchronous,
}

# Define a map to convert test scenario input string into its corresponding enum value
scenario_map = {
    "SingleStream": lg.TestScenario.SingleStream,
    "MultiStream": lg.TestScenario.MultiStream,
    "Server": lg.TestScenario.Server,
    "Offline": lg.TestScenario.Offline,
}


class LoadgenSUT:
    """Loadgen System-Under-Test wrapper."""

    def issue_query(self, query_samples: Iterable[Any]):
        raise NotImplementedError

    def flush_queries(self):
        raise NotImplementedError

    def setup(self, flag_dict: Dict[str, Any], test_settings: lg.TestSettings):
        raise NotImplementedError

    def start(self, flag_dict: Dict[str, Any]):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError


class LoadgenQSL:
    """Loadgen Query-Sample-Library wrapper."""

    def setup(self):
        raise NotImplementedError

    def load_query_samples(self, sample_list: Iterable[int]):
        raise NotImplementedError

    def unload_query_samples(self, sample_list: Iterable[int]):
        raise NotImplementedError


class LoadgenWorkload(Operation):
    """Represents a LoadgenWorkload. Implementations should set up the necessary components for an MLPerf Inference
    workload.
    """
    def __init__(self, flag_dict: Dict[str, Any]):
        self.test_settings = lg.TestSettings()
        self.log_settings = lg.LogSettings()

        # Set common default flags
        flag_dict["gpu_engines"] = flag_dict.get("gpu_engines", "")
        flag_dict["dla_engines"] = flag_dict.get("dla_engines", "")
        flag_dict["plugins"] = flag_dict.get("plugins", "")
        flag_dict["use_graphs"] = flag_dict.get("use_graphs", False)
        flag_dict["verbose"] = flag_dict.get("verbose", False)
        flag_dict["verbose_nvtx"] = flag_dict.get("verbose_nvtx", False)

        flag_dict["scenario"] = flag_dict.get("scenario", "Offline")
        flag_dict["test_mode"] = flag_dict.get("test_mode", "PerformanceOnly")

        # Configuration files
        flag_dict["mlperf_conf_path"] = flag_dict.get("mlperf_conf_path", "")
        flag_dict["user_conf_path"] = flag_dict.get("user_conf_path", "")

        # Loadgen logging settings
        flag_dict["logfile_outdir"] = flag_dict.get("logfile_outdir", "")
        flag_dict["logfile_prefix"] = flag_dict.get("logfile_prefix", "")
        flag_dict["logfile_suffix"] = flag_dict.get("logfile_suffix", "")
        flag_dict["log_mode"] = flag_dict.get("log_mode", "AsyncPoll")
        flag_dict["log_enable_trace"] = flag_dict.get("log_enable_trace", False)

        # QSL arguments
        flag_dict["map_path"] = flag_dict.get("map_path", "")
        flag_dict["tensor_path"] = flag_dict.get("tensor_path", "")
        flag_dict["start_from_device"] = flag_dict.get("start_from_device", False)
        flag_dict["numa_config"] = flag_dict.get("numa_config", "")

        # Set output_keys
        self.benchmark = flag_dict["model"]
        self.scenario = flag_dict["scenario"]
        self.mode = flag_dict["test_mode"]

    def setup(self):
        flag_dict = self.flag_dict
        # Configure the test settings
        self.test_settings.scenario = scenario_map[flag_dict["scenario"]]
        self.test_settings.mode = test_mode_map[flag_dict["test_mode"]]

        logging.info(f'mlperf.conf path: {flag_dict["mlperf_conf_path"]}')
        logging.info(f'user.conf path: {flag_dict["user_conf_path"]}')
        self.test_settings.FromConfig(flag_dict["mlperf_conf_path"], flag_dict["model"], flag_dict["scenario"])
        self.test_settings.FromConfig(flag_dict["user_conf_path"], flag_dict["model"], flag_dict["scenario"])

        # Configure the logging settings
        self.log_settings.log_output.outdir = flag_dict["logfile_outdir"]
        self.log_settings.log_output.prefix = flag_dict["logfile_prefix"]
        self.log_settings.log_output.suffix = flag_dict["logfile_suffix"]
        self.log_settings.log_output.prefix_with_datetime = flag_dict["logfile_prefix_with_datetime"]
        self.log_settings.log_output.copy_detail_to_stdout = flag_dict["log_copy_detail_to_stdout"]
        self.log_settings.log_output.copy_summary_to_stdout = not flag_dict["disable_log_copy_summary_to_stdout"]
        self.log_settings.log_mode = log_mode_map[flag_dict["log_mode"]]
        self.log_settings.log_mode_async_poll_interval_ms = flag_dict["log_mode_async_poll_interval_ms"]
        self.log_settings.enable_trace = flag_dict["log_enable_trace"]

    def run(self):
        raise NotImplementedError

    @classmethod
    def output_keys(cls):
        return ["benchmark",
                "scenario",
                "mode",
                "sut",
                "qsl"]


class LoadgenBenchmark(Operation):
    """Represents a benchmark that utilizes Loadgen to run."""
    # TODO: Provide a default run() method that all LoadgenBenchmarks should utilize.

    @classmethod
    def output_keys(cls):
        return []  # TODO: Figure this out

    @classmethod
    def immediate_dependencies(cls):
        return [LoadgenWorkload]
