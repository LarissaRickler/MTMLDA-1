import logging
import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
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
class EntryType(Enum):
    BASIC = 0
    RUNNING = 1
    ACCUMULATIVE = 2


# --------------------------------------------------------------------------------------------------
@dataclass
class StatisticsEntry:
    str_format: str = None
    str_name: str = None


@dataclass
class BasicStatisticsEntry(StatisticsEntry):
    value: Any = None


@dataclass
class RunningStatisticsEntry(StatisticsEntry):
    value: list[Any] = ()


@dataclass
class AccumulativeStatisticsEntry(StatisticsEntry):
    value: list[Any] = ()
    num_recordings: int = 0
    average: float = 0


# ==================================================================================================
class Statistics:
    # ----------------------------------------------------------------------------------------------
    def __init__(self) -> None:
        self._entries = {}

    # ----------------------------------------------------------------------------------------------
    def add_entry(self, identifier, type, str_format, str_name=None):
        if type == EntryType.BASIC:
            self._entries[identifier] = BasicStatisticsEntry(
                str_format=str_format, str_name=str_name
            )
        elif type == EntryType.RUNNING:
            self._entries[identifier] = RunningStatisticsEntry(
                str_format=str_format, str_name=str_name
            )
        elif type == EntryType.ACCUMULATIVE:
            self._entries[identifier] = AccumulativeStatisticsEntry(
                str_format=str_format, str_name=str_name
            )

    # ----------------------------------------------------------------------------------------------
    def set_value(self, identifier, value):
        entry = self._entries[identifier]
        if isinstance(entry, BasicStatisticsEntry):
            self._entries[identifier].value = value
        elif isinstance(entry, (RunningStatisticsEntry, AccumulativeStatisticsEntry)):
            self._entries[identifier].value.append(value)

    # ----------------------------------------------------------------------------------------------
    def get_statistics(self):
        for entry_id in self._entries.keys():
            entry = self._entries[entry_id]
            if isinstance(entry, BasicStatisticsEntry):
                value = entry.value
            elif isinstance(entry, RunningStatisticsEntry):
                value = np.column_stack(entry.value)
                value = np.mean(entry.value, axis=-1)
                self._entries[entry_id].value = ()
            elif isinstance(entry, AccumulativeStatisticsEntry):
                value = np.column_stack(entry.value)
                new_average = np.mean(entry.value, axis=-1)
                num_new_recordings = len(entry.value)
                record_ratio = num_new_recordings / (num_new_recordings + entry.num_recordings)
                value = record_ratio * new_average + (1 - record_ratio) * entry.average
                self._entries[entry_id].average = value
                self._entries[entry_id].num_recordings += num_new_recordings
                self._entries[entry_id].value = ()

            yield entry_id, value, entry.str_format

    # ----------------------------------------------------------------------------------------------
    @property
    def entries(self):
        return self._entries


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
    def log_run_statistics(self, statistics: Statistics) -> None:
        output_str = ""

        for _, value, entry in statistics.get_entries():
            value_str = self._process_value_str(value, entry)
            output_str += f"{value_str}| "
        self.info(output_str)

    # ----------------------------------------------------------------------------------------------
    def log_debug_statistics(self, statistics: Statistics, info) -> None:
        output_str = ""
        for identifier, value, entry in statistics.get_entries():
            value_str = self._process_value_str(value, entry)
            output_str += f"{identifier}: {value_str}| "

        info_str = f"[{info}]"
        output_str = f"{info_str:15} {output_str}"
        self.debug(output_str)

    # ----------------------------------------------------------------------------------------------
    def print_log_header(self, statistics: Statistics) -> None:
        log_header_str = ""
        for entry in statistics.entries:
            log_header_str += entry.str_name
        self.info(log_header_str)
        self.info("-" * (len(log_header_str) - 1))

    # ----------------------------------------------------------------------------------------------
    def print_debug_new_samples(self, sample: int) -> None:
        if self._debugfile_path is not None:
            output_str = f" New chain segment, sample {sample:<8.3e} ".center(
                self._debug_header_width, "="
            )
            self.debug(f"\n{output_str}\n")

    # ----------------------------------------------------------------------------------------------
    def print_debug_tree_export(self, tree_id: int) -> None:
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
            value_str = f"({",".join(value_str)})"
        elif value is None:
            value_str = np.nan
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
