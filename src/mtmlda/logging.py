import logging
import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


# ==================================================================================================
@dataclass
class LoggerSettings:
    do_printing: bool = True
    logfile_path: str = None
    debugfile_path: str = None
    write_mode: str = "w"


# ==================================================================================================
class Statistic:
    def __init__(self, str_id, str_format):
        self.str_id = str_id
        self.str_format = str_format
        self._value: Any = None

    def set_value(self, value):
        self._value = value

    def get_value(self):
        return self._value

# --------------------------------------------------------------------------------------------------
class RunningStatistic(Statistic):
    def __init__(self, str_id, str_format):
        super().__init__(str_id, str_format)
        self._value = []

    def set_value(self, new_value):
        self.value.append(new_value)

    def get_value(self):
        value = np.column_stack(self._value)
        value = np.mean(value, axis=-1)
        self._value = []
        return value

# --------------------------------------------------------------------------------------------------
class AccumulativeStatistic(Statistic):
    def __init__(self, str_id, str_format):
        super().__init__(str_id, str_format)
        self._value = []
        self._num_recordings = 0
        self._average = 0

    def set_value(self, new_value):
        self.value.append(new_value)

    def get_value(self):
        value = np.column_stack(self._value)
        num_new_recordings = len(self._value)
        new_average = np.mean(value, axis=-1)
        record_ratio = num_new_recordings / (num_new_recordings + self._num_recordings)
        value = record_ratio * new_average + (1 - record_ratio) * self._average
        self._average = value
        self._num_recordings += num_new_recordings
        self._value = ()
        return value


# ==================================================================================================
class MTMLDALogger:
    _debug_header_width = 80

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        logger_settings: LoggerSettings,
    ) -> None:
        self._logfile_path = logger_settings.logfile_path
        self._debugfile_path = logger_settings.debugfile_path
        self._pylogger = logging.getLogger(__name__)
        self._pylogger.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(message)s")

        if not self._pylogger.hasHandlers():
            if logger_settings.do_printing:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setFormatter
                console_handler.setLevel(logging.INFO)
                console_handler.setFormatter(formatter)
                self._pylogger.addHandler(console_handler)

            if self._logfile_path is not None:
                os.makedirs(self._logfile_path.parent, exist_ok=True)
                file_handler = logging.FileHandler(
                    self._logfile_path, mode=logger_settings.write_mode
                )
                file_handler.setFormatter(formatter)
                file_handler.setLevel(logging.INFO)
                self._pylogger.addHandler(file_handler)

            if self._debugfile_path is not None:
                os.makedirs(self._debugfile_path.parent, exist_ok=True)
                debug_handler = DebugFileHandler(
                    self._debugfile_path, mode=logger_settings.write_mode
                )
                debug_handler.setFormatter(formatter)
                debug_handler.setLevel(logging.DEBUG)
                self._pylogger.addHandler(debug_handler)

    # ----------------------------------------------------------------------------------------------
    def log_run_statistics(self, statistics: dict[str, Statistic]) -> None:
        output_str = ""

        for statistic in statistics.values():
            value_str = self._process_value_str(statistic.get_value(), statistic.str_format)
            output_str += f"{value_str}| "
        self.info(output_str)

    # ----------------------------------------------------------------------------------------------
    def log_debug_statistics(self, info: str, statistics: dict[str, Statistic]) -> None:
        output_str = ""
        for statistic in statistics.values():
            value_str = self._process_value_str(statistic.get_value(), statistic.str_format)
            output_str += f"{statistic.str_id}: {value_str}| "

        info_str = f"[{info}]"
        output_str = f"{info_str:15} {output_str}"
        self.debug(output_str)

    # ----------------------------------------------------------------------------------------------
    def log_header(self, statistics: dict[str, Statistic]) -> None:
        log_header_str = ""
        for statistic in statistics.values():
            log_header_str += f"{statistic.str_id}| "
        self.info(log_header_str)
        self.info("-" * (len(log_header_str) - 1))

    # ----------------------------------------------------------------------------------------------
    def log_debug_new_samples(self, sample: int) -> None:
        if self._debugfile_path is not None:
            output_str = f" New chain segment, sample {sample:<8.3e} ".center(
                self._debug_header_width, "="
            )
            self.debug(f"\n{output_str}\n")

    # ----------------------------------------------------------------------------------------------
    def log_debug_tree_export(self, tree_id: int) -> None:
        if (self._debugfile_path is not None) and (tree_id is not None):
            output_str = f"-> Export tree with Id {tree_id}"
            self.debug(output_str)

    # ----------------------------------------------------------------------------------------------
    def info(self, message: str) -> None:
        self._pylogger.info(message)

    # ----------------------------------------------------------------------------------------------
    def debug(self, message: str) -> None:
        self._pylogger.debug(message)

    # ----------------------------------------------------------------------------------------------
    def exception(self, message: str) -> None:
        self._pylogger.exception(message)

    # ----------------------------------------------------------------------------------------------
    def _process_value_str(self, value: Any, str_format: str) -> str:
        if isinstance(value, Iterable):
            value_str = [f"{val:{str_format}}" for val in value]
            value_str = f"({','.join(value_str)})"
        elif value is None:
            value_str = f"{np.nan:{str_format}}"
        else:
            value_str = f"{value:{str_format}}"
        return value_str


# ==================================================================================================
class DebugFileHandler(logging.FileHandler):
    def __init__(self, filename: Path, mode: str = "a", encoding: str = None, delay: bool = False):
        super().__init__(filename, mode, encoding, delay)

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno == logging.DEBUG:
            super().emit(record)
