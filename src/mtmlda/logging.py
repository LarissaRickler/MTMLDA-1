import logging
import os
import sys

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import mltree


# ==================================================================================================
@dataclass
class LoggerSettings:
    do_printing: bool
    logfile_path: str
    debugfile_path: str
    write_mode: str


# ==================================================================================================
class MTMLDALogger:
    _debug_header_width = 80

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        logger_settings: LoggerSettings,
        components: dict[str, Any],
    ) -> None:
        self._logfile_path = logger_settings.logfile_path
        self._debugfile_path = logger_settings.debugfile_path
        self._components = components
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
    def print_headers(self) -> None:
        self._print_log_header()
        if self._debugfile_path is not None:
            self._print_debug_header()

    # ----------------------------------------------------------------------------------------------
    def print_statistics(self, info_dict: dict[str, Any]) -> None:
        output_str = ""

        for component, value in info_dict.items():
            component_format = self._components[component]["format"]
            output_str += f"{value:<{component_format}}| "
        self._pylogger.info(output_str)

    # ----------------------------------------------------------------------------------------------
    def print_debug_info(self, info: str, node: mltree.MTNode) -> None:
        if self._debugfile_path is not None:
            state_str = [f"{state:<12.3e}" for state in np.nditer(node.state)]
            state_str = ",".join(state_str)
            if node.logposterior is None:
                logp_str = f"{'None':12}"
            else:
                logp_str = f"{node.logposterior:<12.3e}"

            node_str = (
                f"L: {node.level:<3} | "
                f"I: {node.subchain_index:<3} | "
                f"S: ({state_str}) | "
                f"P: {logp_str} | "
                f"R: {node.probability_reached:<12.3e}"
            )
            info_str = f"[{info}]"
            output_str = f"{info_str:15} {node_str}"
            self.debug(output_str)

    def print_debug_new_samples(self, sample: int) -> None:
        if self._debugfile_path is not None:
            output_str = f" New chain segment, sample {sample:<8.3e} ".center(
                self._debug_header_width, "="
            )
            self.debug(f"\n{output_str}\n")

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
    def _print_log_header(self) -> None:
        log_header_str = ""
        for component in self._components.keys():
            component_name = self._components[component]["id"]
            component_width = self._components[component]["width"]
            log_header_str += f"{component_name:{component_width}}| "
        separator = "-" * (len(log_header_str) - 1)
        self._pylogger.info(log_header_str)
        self._pylogger.info(separator)

    # ----------------------------------------------------------------------------------------------
    def _print_debug_header(self) -> None:
        debug_header_str = (
            "Explanation of abbreviations:\n\n"
            "L: level\n"
            "I: subchain index\n"
            "S: state\n"
            "P: log posterior\n"
            "R: probability reached\n"
        )
        self.debug(debug_header_str)
        self.print_debug_new_samples(sample=1)


# ==================================================================================================
class DebugFileHandler(logging.FileHandler):
    def __init__(self, filename: Path, mode: str = "a", encoding: str = None, delay: bool = False):
        super().__init__(filename, mode, encoding, delay)

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno == logging.DEBUG:
            super().emit(record)
