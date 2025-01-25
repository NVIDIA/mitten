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
from collections import UserDict
from inspect import isclass

import enum
import importlib
import json
import logging


class JSONable:

    _registered = dict()

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        JSONable._registered[str(cls)] = cls

    def json_encode(self):
        raise NotImplemented

    @classmethod
    def from_json(cls, d):
        raise NotImplemented


class Encoder(json.JSONEncoder):

    def default(self, obj):
        if str(obj.__class__) in JSONable._registered:
            return {"__obj_class__": str(obj.__class__),
                    "data": obj.json_encode()}
        elif isinstance(obj, enum.Enum):
            return {"__ENUM__": [obj.__class__.__module__, obj.__class__.__name__],
                    "name": obj.name}
        elif isclass(obj):
            return {"__CLASS__": [obj.__module__, obj.__name__]}
        elif isinstance(obj, set):
            return {"__SET__": list(obj)}
        return json.JSONEncoder.default(self, obj)


class Decoder(json.JSONDecoder):

    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, d):
        if type(d) is not dict:
            return d

        if "__obj_class__" in d:
            k = d["__obj_class__"]
            if k not in JSONable._registered:
                logging.warn(f"Class {k} is in Mitten JSONable format, but not registered as JSONable object")
                return d
            return JSONable._registered[k].from_json(d["data"])
        elif "__ENUM__" in d:
            _mod, _name = d["__ENUM__"]
            e = getattr(importlib.import_module(_mod), _name)
            return getattr(e, d["name"])
        elif "__CLASS__" in d:
            _mod, _name = d["__CLASS__"]
            return getattr(importlib.import_module(_mod), _name)
        elif "__SET__" in d:
            return set(d["__SET__"])
        return d


class ClassKeyedDict(JSONable, UserDict):
    """Utility class for a common pattern in Mitten where classes are keyed with classes.
    """

    def __setitem__(self, k, v):
        if not isclass(k):
            raise KeyError(f"ClassKeyedDict key '{k}' must be a class")
        super().__setitem__(k, v)

    def json_encode(self):
        return [{"key.module": k.__module__,
                 "key.name": k.__name__,
                 "value": v}
                for k, v in self.items()]

    @classmethod
    def from_json(cls, L):
        decoded = cls()
        for elem in L:
            k = getattr(importlib.import_module(elem["key.module"]), elem["key.name"])
            decoded[k] = elem["value"]
        return decoded


def dump(*args, **kwargs):
    return json.dump(*args, cls=Encoder, **kwargs)


def dumps(*args, **kwargs):
    return json.dumps(*args, cls=Encoder, **kwargs)


def load(*args, **kwargs):
    return json.load(*args, cls=Decoder, **kwargs)


def loads(*args, **kwargs):
    return json.loads(*args, cls=Decoder, **kwargs)
