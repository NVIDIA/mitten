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


from pathlib import Path

import datetime
import io
import numpy as np
import pickle
import pytest
import sys

from nvmitten.debug.debug_manager import DebugManager, DebuggableMixin, EventRecord


def test_event_record_no_message():
    rec = EventRecord(datetime.datetime.utcfromtimestamp(0),
                      "test_event_record",
                      "sample_name")

    assert str(rec) == "[1970-01-01T00:00:00] (test_event_record) sample_name"
    assert EventRecord.from_json(rec.to_json()) == rec


def test_event_record_with_message():
    rec = EventRecord(datetime.datetime.utcfromtimestamp(0),
                      "test_event_record",
                      "sample_name",
                      message="hello\nworld")

    assert str(rec) == "[1970-01-01T00:00:00] (test_event_record) sample_name: hello\nworld"
    assert EventRecord.from_json(rec.to_json()) == rec


def test_event_record_convert_exc_info():
    exc_info = None
    try:
        raise NotImplementedError
    except NotImplementedError:
        exc_info = sys.exc_info()
    assert exc_info is not None

    d = EventRecord.convert_exc_info(exc_info)
    assert d["type"] == "builtins.NotImplementedError"
    assert isinstance(d["traceback"], str)
    tb_lines = d["traceback"].split("\n")
    assert tb_lines[-1] != ""

    # TODO: Check the actual lines are in the right format. For now, it's probably safe to rely on the traceback module.


def test_event_record_with_stackinfo():
    rec = EventRecord(datetime.datetime.utcfromtimestamp(0),
                      "test_event_record",
                      "sample_name",
                      message="hello\nworld",
                      stack_info="this is stack depth 1\nthis is stack depth 2")

    assert str(rec) == "[1970-01-01T00:00:00] (test_event_record) sample_name: hello\nworld\n\n" \
                       "-- STACK INFO BEGIN --\nthis is stack depth 1\nthis is stack depth 2\n-- STACK INFO END --"
    assert EventRecord.from_json(rec.to_json()) == rec


def test_debug_manager_write_log_line(debug_manager_io_stream):
    rec1 = EventRecord(datetime.datetime.utcfromtimestamp(0),
                       "test_debug_manager_write_log_line",
                       "sample_name")
    DebugManager._write_log_line(rec1)
    contents = debug_manager_io_stream.getvalue().split('\n')
    assert len(contents) == 2
    assert EventRecord.from_json(contents[0]) == rec1
    assert contents[-1] == ""

    rec2 = EventRecord(datetime.datetime.utcfromtimestamp(60 * 60 * 24),
                       "test_debug_manager_write_log_line",
                       "sample_name",
                       message="hello\nworld")
    DebugManager._write_log_line(rec2)
    contents = debug_manager_io_stream.getvalue().split('\n')
    assert len(contents) == 3
    assert EventRecord.from_json(contents[0]) == rec1
    assert EventRecord.from_json(contents[1]) == rec2
    assert contents[-1] == ""


def test_debug_manager_log_event(debug_manager_io_stream):
    DebugManager.log_event("test_debug_manager_log_event",
                           "test_name",
                           message="my message",
                           timestamp=datetime.datetime.utcfromtimestamp(0))
    contents = debug_manager_io_stream.getvalue().split('\n')
    assert len(contents) == 2
    assert EventRecord.from_json(contents[0]) == EventRecord(datetime.datetime.utcfromtimestamp(0),
                                                             "test_debug_manager_log_event",
                                                             "test_name",
                                                             message="my message")
    assert contents[-1] == ""


def test_debug_manager_register_event(debug_manager_io_stream):
    @DebugManager.register_event
    def fun1(a, b, c=True):
        return a + b if c else a - b

    assert fun1(1, 2) == 3
    assert fun1(1, 2, c=False) == -1

    contents = debug_manager_io_stream.getvalue().split('\n')
    assert len(contents) == (2 * 2) + 1  # 2 per call, +1 newline
    assert contents[-1] == ""

    begin_event1 = EventRecord.from_json(contents[0])
    assert begin_event1.caller.split('.')[-1] == "fun1"
    assert begin_event1.name == "begin"
    assert begin_event1.stack_info == None

    end_event1 = EventRecord.from_json(contents[1])
    assert end_event1.caller.split('.')[-1] == "fun1"
    assert end_event1.name == "end"
    assert end_event1.stack_info == None

    begin_event2 = EventRecord.from_json(contents[2])
    assert begin_event2.caller.split('.')[-1] == "fun1"
    assert begin_event2.name == "begin"
    assert begin_event2.stack_info == None

    end_event2 = EventRecord.from_json(contents[3])
    assert end_event2.caller.split('.')[-1] == "fun1"
    assert end_event2.name == "end"
    assert end_event2.stack_info == None


def test_debug_manager_register_event_exception(debug_manager_io_stream):
    @DebugManager.register_event
    def fun_should_raise_error(a, b):
        raise NotImplementedError

    with pytest.raises(NotImplementedError):
        fun_should_raise_error(1, 2)


    contents = debug_manager_io_stream.getvalue().split('\n')
    assert len(contents) == (2 * 1) + 1  # 2 per call, +1 newline
    assert contents[-1] == ""

    begin_event1 = EventRecord.from_json(contents[0])
    assert begin_event1.caller.split('.')[-1] == "fun_should_raise_error"
    assert begin_event1.name == "begin"
    assert begin_event1.exc_info == None
    assert begin_event1.stack_info == None

    end_event1 = EventRecord.from_json(contents[1])
    assert end_event1.caller.split('.')[-1] == "fun_should_raise_error"
    assert end_event1.name == "end"
    assert end_event1.exc_info is not None
    assert end_event1.exc_info["type"] == "builtins.NotImplementedError"
    assert end_event1.stack_info == None


def test_dump_debug_artifact(debug_manager_io_stream):
    d = {"a": 1, "100": 100}
    DebugManager.dump_debug_artifact("test_dump_debug_artifact",
                                     "dictionary",
                                     d)

    e = np.array([1, 2, 3, 4])
    b = pickle.dumps(e)
    DebugManager.dump_debug_artifact("test_dump_debug_artifact",
                                     "np_ndarray",
                                     b,
                                     obj_already_serialized=True)

    contents = debug_manager_io_stream.getvalue().split('\n')
    assert len(contents) == (2 * 1) + 1  # 2 per call, +1 newline
    assert contents[-1] == ""

    # Verify first dump was logged
    dump_event1 = EventRecord.from_json(contents[0])
    assert dump_event1.caller.split('.')[-1] == "test_dump_debug_artifact"
    assert dump_event1.name == "dictionary"
    assert dump_event1.exc_info == None
    assert dump_event1.stack_info == None
    artifact_path1 = DebugManager.artifacts_dir_path / Path(dump_event1.message.split(": ")[-1])
    assert artifact_path1.is_file()

    # Verify second dump was logged
    dump_event2 = EventRecord.from_json(contents[1])
    assert dump_event2.caller.split('.')[-1] == "test_dump_debug_artifact"
    assert dump_event2.name == "np_ndarray"
    assert dump_event2.exc_info == None
    assert dump_event2.stack_info == None
    artifact_path2 = DebugManager.artifacts_dir_path / Path(dump_event2.message.split(": ")[-1])
    assert artifact_path2.is_file()

    # Verify dump locations are unique and are in the same directory
    assert artifact_path1 != artifact_path2
    assert artifact_path1.parent == artifact_path2.parent
    assert artifact_path1.parent == DebugManager.artifacts_dir_path

    # Verify dumped objects are identical
    with artifact_path1.open(mode='rb') as f:
        d_recovered = pickle.load(f)
    assert d_recovered == d

    with artifact_path2.open(mode='rb') as f:
        e_recovered = pickle.load(f)
    assert (e_recovered == e).all()


def test_debuggable_mixin_normal(debug_manager_io_stream):
    class Foo(DebuggableMixin):
        def __init__(self, a, b, c=2):
            self.a = a
            self.b = b
            self.c = c

        def f1(self, a):
            return self.a + a

    args = (2, {"field1": "value1", "field2": True, "field3": 17})
    kwargs = {"c": 27}
    f = Foo(*args, **kwargs)
    res = f.f1(2)
    assert res == 4

    contents = debug_manager_io_stream.getvalue().split('\n')
    # Should have 5 total events: __init__ call args dump, __init__ begin and end, f1 begin and end.
    assert len(contents) == (5 * 1) + 1  # 2 per call, +1 newline
    assert contents[-1] == ""

    get_caller = lambda ev: '.'.join(ev.caller.split('.')[-2:])

    event1 = EventRecord.from_json(contents[0])
    assert get_caller(event1) == "Foo.__init__"
    assert event1.name == "call_args"
    assert event1.exc_info == None
    assert event1.stack_info == None
    assert event1.message.startswith("Dumping artifact to:")
    p = DebugManager.artifacts_dir_path / Path(event1.message.split(": ")[-1])
    assert p.is_file()
    with p.open(mode='rb') as f:
        recovered = pickle.load(f)
    assert recovered["args"] == args
    assert recovered["kwargs"] == kwargs

    event2 = EventRecord.from_json(contents[1])
    assert get_caller(event2) == "Foo.__init__"
    assert event2.name == "begin"
    assert event2.exc_info == None
    assert event2.stack_info == None

    event3 = EventRecord.from_json(contents[2])
    assert get_caller(event3) == "Foo.__init__"
    assert event3.name == "end"
    assert event3.exc_info == None
    assert event3.stack_info == None

    event4 = EventRecord.from_json(contents[3])
    assert get_caller(event4) == "Foo.f1"
    assert event4.name == "begin"
    assert event4.exc_info == None
    assert event4.stack_info == None

    event5 = EventRecord.from_json(contents[4])
    assert get_caller(event5) == "Foo.f1"
    assert event5.name == "end"
    assert event5.exc_info == None
    assert event5.stack_info == None
