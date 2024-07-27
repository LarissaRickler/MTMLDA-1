"""Custom Logging.

This module provides extensions of Pythons's logging capabilities for run and debug logs within 
MLDA runs. The more elaborate logging routines take `Statistics` objects, which makes their
evaluation and formatted output more convenient.

Classes:
    LoggerSettings: Data class storing settings for the logger
    Statistic: Basic statistics object containing information for logging
    RunningStatistic: Statistic for computing batch averages
    AccumulativeStatistic: Statistic for computing overall averages
    MTMLDALogger: Logger for the MLDA sampler
    DebugFileHandler: Custom file handler for debug logging
"""

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
    """Data class storing settings for the logger.

    Attributes:
        do_printing (bool): Decides if the default logger prints to the console, default is True
        logfile_path (str): Directory to log run statistics to, default is None
        debugfile_path (str): Directory to log debug statistics to, default is None
        write_mode (str): Write mode for the log files (append, overwrite), default is 'w'
    """
    do_printing: bool = True
    logfile_path: str = None
    debugfile_path: str = None
    write_mode: str = "w"


# ==================================================================================================
class Statistic:
    """Basic statistics object containing information for logging.
    
    Every statistics object has a string identifier and a format string for the value it stores. The
    value attribute is not accessed directly, but through getter and setter methods. The idea behind
    is that more sophisticated logic can be implemented in subclasses, still adhering to the generic
    interface.

    Attributes:
        str_id (str): Identifier for the statistic
        str_format (str): Format string for the value

    Methods:
        set_value: Set the value of the statistic
        get_value: Get the value of the statistic
    """

    def __init__(self, str_id, str_format):
        """Constructor of the statistic.

        Args:
            str_id (_type_): Identifier for the statistic
            str_format (_type_): Format string for the value
        """
        self.str_id = str_id
        self.str_format = str_format
        self._value: Any = None

    def set_value(self, value):
        assert isinstance(value, (int, float, np.ndarray)), "Unsupported type for value"
        """Set the value of the statistic."""
        self._value = value

    def get_value(self):
        """Get the value of the statistic."""
        return self._value

# --------------------------------------------------------------------------------------------------
class RunningStatistic(Statistic):
    """Statistic for computing batch averages.
    
    The running statistics stores all values provided by the `set_value` method and computes their
    average when the `get_value` method is called.
    """

    def __init__(self, str_id, str_format):
        """Constructor, see base class for details."""
        super().__init__(str_id, str_format)
        self._value = []

    def set_value(self, new_value):
        """Set the value of the statistic."""
        assert isinstance(new_value, (int, float, np.ndarray)), "Unsupported type for value"
        self.value.append(new_value)

    def get_value(self):
        """Get the value of the statistic."""
        value = np.column_stack(self._value)
        value = np.mean(value, axis=-1)
        self._value = []
        return value

# --------------------------------------------------------------------------------------------------
class AccumulativeStatistic(Statistic):
    """Statistic for computing overall averages.
    
    The mean value is computed from all provided values.
    """
    def __init__(self, str_id, str_format):
        """Constructor, see base class for details."""
        super().__init__(str_id, str_format)
        self._value = []
        self._num_recordings = 0
        self._average = 0

    def set_value(self, new_value):
        """Set the value of the statistic."""
        assert isinstance(new_value, (int, float, np.ndarray)), "Unsupported type for value"
        self.value.append(new_value)

    def get_value(self):
        """Get the value of the statistic's running average."""
        value = np.column_stack(self._value)
        num_new_recordings = len(self._value)
        new_average = np.mean(value, axis=-1)
        record_ratio = num_new_recordings / (num_new_recordings + self._num_recordings)
        value = record_ratio * new_average + (1 - record_ratio) * self._average
        self._average = value
        self._num_recordings += num_new_recordings
        self._value =  []
        return value


# ==================================================================================================
class MTMLDALogger:
    """Logger for the MLDA sampler.
    
    This custom logger wraps the standard Python logger, adding some convenience methods for
    logging on different levels to different files and the console.

    Methods:
        log_run_statistics: Log run statistics
        log_debug_statistics: Log debug statistics
        log_header: Log the header of the run statistics table
        log_debug_new_samples: Log the start of a new chain segment in the debug log
        log_debug_tree_export: Log the export of a tree in the debug log
        info: Log an info message to the run logger
        debug: Log a debug message to the debug logger
        exception: Log an exception to all loggers
    """
    _debug_header_width = 80

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        logger_settings: LoggerSettings,
    ) -> None:
        """Constructor of the logger.

        Initializes the run and debug log handles, depending on the user settings.

        Args:
            logger_settings (LoggerSettings): User settings
        """
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
        """Log statistics into run log table.

        Args:
            statistics (dict[str, Statistic]): Run statistics object to log
        """
        output_str = ""

        for statistic in statistics.values():
            value_str = self._process_value_str(statistic.get_value(), statistic.str_format)
            output_str += f"{value_str}| "
        self.info(output_str)

    # ----------------------------------------------------------------------------------------------
    def log_debug_statistics(self, info: str, statistics: dict[str, Statistic]) -> None:
        """Log statistics into debug file.

        Args:
            info (str): Event to log statistics for
            statistics (dict[str, Statistic]): Statistics to log
        """
        output_str = ""
        for statistic in statistics.values():
            value_str = self._process_value_str(statistic.get_value(), statistic.str_format)
            output_str += f"{statistic.str_id}: {value_str}| "

        info_str = f"[{info}]"
        output_str = f"{info_str:15} {output_str}"
        self.debug(output_str)

    # ----------------------------------------------------------------------------------------------
    def log_header(self, statistics: dict[str, Statistic]) -> None:
        """Print our the header for the run statistics table.

        Args:
            statistics (dict[str, Statistic]): Statistics to print header for
        """
        log_header_str = ""
        for statistic in statistics.values():
            log_header_str += f"{statistic.str_id}| "
        self.info(log_header_str)
        self.info("-" * (len(log_header_str) - 1))

    # ----------------------------------------------------------------------------------------------
    def log_debug_new_samples(self, sample: int) -> None:
        """Log divider into debug file, indicating new fine-level sample.

        Args:
            sample (int): Number of the new sample
        """
        if self._debugfile_path is not None:
            output_str = f" New chain segment, sample {sample:<8.3e} ".center(
                self._debug_header_width, "="
            )
            self.debug(f"\n{output_str}\n")

    # ----------------------------------------------------------------------------------------------
    def log_debug_tree_export(self, tree_id: int) -> None:
        """Note in debug file that Markov tree with given id has been exported.

        Args:
            tree_id (int): ID of the tree that has been exported
        """
        if (self._debugfile_path is not None) and (tree_id is not None):
            output_str = f"-> Export tree with Id {tree_id}"
            self.debug(output_str)

    # ----------------------------------------------------------------------------------------------
    def info(self, message: str) -> None:
        """Wrapper for Python logger info call.

        Args:
            message (str): Info message to log
        """
        self._pylogger.info(message)

    # ----------------------------------------------------------------------------------------------
    def debug(self, message: str) -> None:
        """Wrapper for Python logger debug call.

        Args:
            message (str): Debug message to log
        """
        self._pylogger.debug(message)

    # ----------------------------------------------------------------------------------------------
    def exception(self, message: str) -> None:
        """Wrapper for Python logger exception call.

        Args:
            message (str): Exception message to log
        """
        self._pylogger.exception(message)

    # ----------------------------------------------------------------------------------------------
    def _process_value_str(self, value: Any, str_format: str) -> str:
        """Format a numerical value as string, given a suitable format.

        if the provided value is `None`, it is formatted as `np.nan`. If the value is iterable,
        all values are concatenated as comma-separated list, each with the provided format.

        Args:
            value (Any): Value to format as string
            str_format (str): format to use

        Raises:
            TypeError: If the value is of an unsupported type

        Returns:
            str: Value as formatted string
        """
        if isinstance(value, Iterable):
            value_str = [f"{val:{str_format}}" for val in value]
            value_str = f"({','.join(value_str)})"
        elif value is None:
            value_str = f"{np.nan:{str_format}}"
        elif isinstance(value, float):
            value_str = f"{value:{str_format}}"
        else:
            raise TypeError(f"Unsupported type for value: {type(value)}")
        
        return value_str


# ==================================================================================================
class DebugFileHandler(logging.FileHandler):
    """Custom file handler for Logger.
    
    This file handler only transfers messages on the `DEBUG`level of python logging.
    """

    def __init__(self, filename: Path, mode: str = "a", encoding: str = None, delay: bool = False):
        """Constructor of the file handler.

        Args:
            filename (Path): File to log to
            mode (str, optional): Write mode for log messages. Defaults to "a".
            encoding (str, optional): Special encoding for messages. Defaults to None.
            delay (bool, optional): Determines if file opening is deferred until first `emit` call.
                Defaults to False.
        """
        super().__init__(filename, mode, encoding, delay)

    def emit(self, record: logging.LogRecord) -> None:
        """Transfer a log message.

        Args:
            record (logging.LogRecord): Log message object 
        """
        if record.levelno == logging.DEBUG:
            super().emit(record)
