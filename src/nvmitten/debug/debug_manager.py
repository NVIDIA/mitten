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

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Dict, Optional, Tuple, Type, Union

import datetime
import inspect
import io
import json
import logging
import os
import pickle
import sys
import tempfile
import traceback
import uuid


EXC_INFO_T = Tuple[Type[Exception], Exception, TracebackType]


@dataclass(eq=True, frozen=True)
class EventRecord:
    timestamp: datetime.datetime
    caller: str
    name: str
    message: Optional[str] = None
    exc_info: Optional[Dict[str, str]] = None
    stack_info: Optional[str] = None

    def to_json(self):
        d = {"time": self.timestamp.isoformat(),
             "caller": self.caller,
             "name": self.name}

        if self.message:
            d["message"] = self.message

        if self.exc_info:
            d["exception"] = self.exc_info

        if self.stack_info:
            d["call_stack"] = self.stack_info

        return json.dumps(d)

    @staticmethod
    def convert_exc_info(exc_info: EXC_INFO_T):
        return {"type": ".".join([exc_info[0].__module__, exc_info[0].__qualname__]),
                "traceback": ''.join(traceback.format_tb(exc_info[2])).rstrip()}

    @classmethod
    def from_json(self, s: str):
        d = json.loads(s)
        return EventRecord(datetime.datetime.fromisoformat(d["time"]),
                           d["caller"],
                           d["name"],
                           message=d.get("message", None),
                           exc_info=d.get("exception", None),
                           stack_info=d.get("call_stack", None))

    def __str__(self):
        s = f"[{self.timestamp.isoformat()}] ({self.caller}) {self.name}"
        if self.message:
            s += f": {self.message}"

        if self.exc_info:
            s = '\n'.join([s,
                           "",
                           f"{self.exc_info['type']} Raised:",
                           "-- EXCEPTION BEGIN --",
                           self.exc_info["traceback"],
                           "-- EXCEPTION END --"])

        if self.stack_info:
            s = '\n'.join([s,
                           "",
                           "-- STACK INFO BEGIN --",
                           self.stack_info,
                           "-- STACK INFO END --"])
        return s


class _DebugManager:
    """Flexible logger to keep track of Mitten operations and pipelines to assist with debugging and diagnosing errors
    that occur during a Mitten pipeline run.

    DebugManager tracks the following:
        1. Raw messages, similar to things like `logging.info`.
        2. Mitten events - named events that can be registered, which dump a log message, as well as optionally a stack
        trace and/or local variable snapshot if verbose or debug mode is enabled.

    DebugManager provides several ways for a Python object to log or dump artifacts:
        1. Bare methods. These are `.dump_debug_artifact` and `.log_event`.
        2. The `@DebugManager.register_event` decorator can be applied to functions. It can also be applied to classes,
        in which case all methods within the class will have this decorator applied.
        3. The `DebuggableMixin` class that can be used as a Mixin class for objects.

    DebugManager can also be integrated into Python's built-in `logging` module by providing a `logging.Handler`
    implementation.
    """

    def __init__(self, log_dir: Optional[os.PathLike] = None):
        """Creates a DebugManager.
        """
        if log_dir is None:
            self.log_dir = Path(tempfile.gettempdir())
        else:
            self.log_dir = Path(log_dir)

        _date_str = datetime.datetime.now().isoformat(timespec="seconds")
        _uuid = uuid.uuid4().hex
        self.session_dir = f"nvmitten_{_date_str}_{_uuid}"

        self.base_dir = self.log_dir / self.session_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.events_log_path = self.base_dir / "events.log"

        self.artifacts_dir_path = self.base_dir / "artifacts"
        self.artifacts_dir_path.mkdir(parents=True)

        self._fd = None
        self._reset_fd()

        self.print_to_stdout = False
        self.log_stack_info_default = False

    def _reset_fd(self, _f: typing.TextIO = None):
        if _f is None:
            _f = self.events_log_path.open(mode='a')
        if self._fd is not None and not self._fd.closed:
            self._fd.close()
        self._fd = _f

    def _write_log_line(self, record: EventRecord, print_to_stdout: Optional[bool] = None):
        """Writes a line to the log file.

        Args:
            record (EventRecord): The EventRecord to write to the log.
            print_to_stdout (bool): If True, also prints the event to stdout. If None, uses class `print_to_stdout`
                                    property. (Default: None)
        """
        # https://stackoverflow.com/a/164770
        if isinstance(self._fd, io.FileIO) and os.fstat(self._fd.fileno()).st_nlink == 0:
            self._reset_fd()

        self._fd.write(record.to_json() + '\n')

        if print_to_stdout is None:
            print_to_stdout = self.print_to_stdout

        if print_to_stdout:
            # TODO: Should this use sys.stdout.write or print???
            print(str(record))

    def log_event(self,
                  caller: str,
                  name: str,
                  message: Optional[str] = None,
                  timestamp: Optional[datetime.datetime] = None,
                  exc_info: Optional[EXC_INFO_T] = None,
                  stack_info: Union[bool, str] = False,
                  print_to_stdout: Optional[bool] = None):
        """Logs an event given a name. The parameters here are the same as `logging.info`.

        Args:
            caller (str): The ID of the creator of the event. Usually <module>.<class>.<function_name>.
            name (str): The name of the event that occurred.
            message (str): Optional message to log. (Default: None)
            timestamp (datetime.datetime): The timestamp of the event. If not provided, uses the time at the time of
                                           logging. (Default: None)
            exc_info (EXC_INFO_T): The exception that occurred as part of this event (if any), given by
                                   `sys.exc_info()`. (Default: None)
            stack_info (bool): If True, additionally will dump the current call stack for the event.
            print_to_stdout (bool): If True, also prints the event to stdout. If None, uses class `print_to_stdout`
                                    property. (Default: None)
        """
        if not timestamp:
            timestamp = datetime.datetime.now()

        if exc_info:
            # Convert the exception info to the proper field
            exc_info = EventRecord.convert_exc_info(exc_info)

        # https://github.com/python/cpython/blob/3.11/Lib/logging/__init__.py#L1587-L1592
        sinfo = None
        if isinstance(stack_info, str):
            sinfo = stack_info
        elif stack_info:
            with io.StringIO() as sio:
                sio.write("Stack (most recent call last):\n")
                traceback.print_stack(f, file=sio)
                sinfo = sio.getvalue()
                if sinfo[-1] == '\n':
                    sinfo = sinfo[:-1]

        record = EventRecord(timestamp,
                             caller,
                             name,
                             message=message,
                             exc_info=exc_info,
                             stack_info=sinfo)

        # TODO: Should there just be a custom JSONEncoder or call `.to_json()` directly??
        self._write_log_line(record, print_to_stdout=print_to_stdout)

    def register_event(self, obj: object):
        """Decorator to register a function as a Mitten event. 2 events will be generated for these function calls:
            1. A `<function>.begin` event signifying that the method was invoked. This event will also dump the call
            arguments to the function.
            2. A `<function>.end` event signifying that the method completed. If an exception was thrown to cause the
            end of the function, `DebugManager.log_event` is called with `exc_info` populated.
        """
        if inspect.ismethod(obj) or inspect.isfunction(obj):
            def _wrapper(*args, **kwargs):
                caller = f"{obj.__module__}.{obj.__qualname__}"
                start_dt = datetime.datetime.now()

                # 'begin' event
                self.log_event(caller,
                               "begin",
                               timestamp=start_dt,
                               stack_info=self.log_stack_info_default)

                # Do operation
                exc_info = None
                retval = None
                try:
                    retval = obj(*args, **kwargs)
                except Exception as exc:
                    exc_info = sys.exc_info()
                finally:
                    # 'end' event
                    self.log_event(caller,
                                   "end",
                                   timestamp=datetime.datetime.now(),
                                   exc_info=exc_info,
                                   stack_info=self.log_stack_info_default)

                if exc_info is not None:
                    # Resurface exception
                    raise exc_info[1]
                else:
                    return retval
            return _wrapper
        else:
            raise TypeError(f"Could not call register_event on non-function: {type(obj)}")

    def log_call_args(self,
                      caller: Optional[str] = None,
                      ignore_first_arg: bool = False):
        """Decorator to log the call arguments of a function.

        If the function is a method in a class (i.e. it is a function within a class, not necessarily a @classmethod),
        then `self` (and `cls` for @classmethods) will also be logged. This provides a dump of the state of the object
        at the point the method was called.

        Args:
            ignore_first_arg (bool): If True, will omit the first element of *args from being dumped. This is useful for
                                     methods where you explicitly do not want to dump `self` / `cls`.
        """
        def decorator(func: object):
            def _wrapper(*args, **kwargs):
                _caller = caller if caller else f"{func.__module__}.{func.__qualname__}"
                timestamp = datetime.datetime.now()

                dump_args = {"args": args[1:] if ignore_first_arg else args,
                             "kwargs": kwargs}
                self.dump_debug_artifact(_caller,
                                         "call_args",
                                         dump_args,
                                         timestamp=timestamp)

                return func(*args, **kwargs)
            return _wrapper
        return decorator

    def dump_debug_artifact(self,
                            caller: str,
                            name: str,
                            obj: object,
                            timestamp: Optional[datetime.datetime] = None,
                            obj_already_serialized: bool = False):
        """Dumps an object as a Pickled debug artifact.

        Args:
            caller (str): The ID of the creator of the event. Usually <module>.<class>.<function_name>.
            name (str): The name of the event that occurred.
            obj (object): The object to dump.
            timestamp (datetime.datetime): The timestamp of the event. If not provided, uses the time at the time of
                                           logging. (Default: None)
            obj_already_serialized (bool): If True, assumes `obj` is the output of `pickle.dumps` and writes the object
                                           directly to the file. Otherwise, serializes and dumps the object. (Default:
                                           False)
        """
        if not timestamp:
            timestamp = datetime.datetime.now()

        isoformat = timestamp.isoformat(timespec="seconds")
        fname = f"{caller}.{name}_{isoformat}.pkl"

        # Log the creation of an artifact
        self.log_event(caller,
                       name,
                       message=f"Dumping artifact to: {fname}",
                       timestamp=timestamp)

        # Write out the artifact
        exc_info = None
        artifact_path = self.artifacts_dir_path / fname
        try:
            with artifact_path.open(mode='wb') as f:
                if obj_already_serialized:
                    f.write(obj)
                else:
                    pickle.dump(obj, f)
        except (pickle.PicklingError, AttributeError) as exc:
            exc_info = sys.exc_info()
            # Log failure
            self.log_event("nvmitten.debug.DebugManager.dump_debug_artifact",
                           "artifact_pickle_error",
                           timestamp=datetime.datetime.now(),
                           exc_info=exc_info,
                           stack_info=self.log_stack_info_default)

            # Delete the file
            artifact_path.unlink(missing_ok=True)

            # Do not re-raise this error


# TODO: Parse args via Fields API to set artifact dir and such.
# Global instance
DebugManager = _DebugManager(log_dir=os.environ.get("NVMITTEN_LOG_DIR", None))


class DebuggableMixin:
    """ Mixin class to automatically add debugging and helper methods to a class. Subclasses of DebuggableMixin will
    automatically have all unbound functions not prefixed by an underscore ('_') decorated by
    `DebugManager.register_event`. This means @classmethods will not be automatically marked as events, and must be
    manually registered.

    `__init__` will also be decorated by `register_event`, as well as
    `DebugManager.log_call_args(ignore_first_arg=True)`.
    """

    @classmethod
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)

        # Search through all attributes
        for attr in dir(cls):
            if not attr.startswith('_') or attr == "__init__":
                if not inspect.isfunction(getattr(cls, attr)):
                    continue

                o = getattr(cls, attr)
                caller = f"{o.__module__}.{o.__qualname__}"

                # Do not touch attributes inherited from Python `object`
                if hasattr(object, attr) and o == getattr(object, attr):
                    continue

                o = DebugManager.register_event(o)
                if attr == "__init__":
                    o = DebugManager.log_call_args(caller=caller, ignore_first_arg=True)(o)
                setattr(cls, attr, o)


class DebugHandler(logging.Handler):
    """Python Logging Facility Handler to enable Python's built-in `logging` module to log to DebugManager as well
    """

    def __init__(self):
        super().__init__(level=logging.NOTSET)  # Defer to the user to set global logging level

    def emit(self, record: logging.LogRecord):
        name = f"{record.name} (Level: {record.levelname})"
        if record.funcName:
            name += f" in {record.funcName}"
            if record.pathname and record.lineno:
                name += f" ({record.pathname}:L{record.lineno})"

        DebugManager.log_event("logging.Logger",
                               name,
                               message=record.message,
                               timestamp=datetime.datetime.fromtimestamp(record.created),
                               exc_info=record.exc_info,
                               stack_info=record.stack_info,
                               print_to_stdout=False)
