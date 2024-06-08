import argparse
import docstring_parser
import inspect
from typing import Iterable
from argparse import ArgumentParser, Action, FileType, Namespace
from collections.abc import Callable
from typing import Any, Container, Iterable, Sequence
from omegaconf import OmegaConf
import argparse
from dataclasses import is_dataclass


def parse_type_or_null(str, type):
    if str == "None":
        return None
    return type(str)


def add_args(
    parser: ArgumentParser, callable, group: str | None = None, ignore_keys=[], **defaults
):
    parsed_doc = docstring_parser.parse(callable.__doc__ or "")
    doc_params = {arg.arg_name: arg.description for arg in parsed_doc.params}

    if group is not None:
        parser = parser.add_argument_group(group, parsed_doc.short_description)

    parameters = inspect.signature(callable).parameters

    for key, param in parameters.items():

        parse_kw = {}

        if key in ignore_keys:
            continue
        
        if key in defaults:
            parse_kw["default"] = defaults[key]
            parse_kw["required"] = False
        else: 
            if param.default is inspect._empty:
                parse_kw["default"] = None
                parse_kw["required"] = True
            else:
                parse_kw["default"] = param.default
                parse_kw["required"] = False

        if param.annotation is inspect._empty:
            raise ValueError(f"Can't add args without annotations")

        if hasattr(param.annotation, "__origin__"):
            if issubclass(param.annotation.__origin__, Iterable):
                parse_kw["nargs"] = "+"
                parse_kw["type"] = param.annotation.__args__[0]
            else:
                raise ValueError(f"Can't parse type annotation {param.annotation}")
        elif hasattr(param.annotation, "__args__") and param.annotation.__args__[1] is type(None):
            parse_kw["type"] = param.annotation.__args__[0]
        else:
            parse_kw["type"] = param.annotation

        parse_kw["help"] = doc_params.get(key, key)

        if parse_kw["type"] is type(True):
            parse_kw.pop("type") 
            parser.add_argument(f"--{key}", **parse_kw, action='store_true')
            if parse_kw["default"] is True:
                parser.add_argument(f"--no-{key}", **parse_kw, action='store_false', dest=key)
        else:
            parser.add_argument(f"--{key}", **parse_kw)


def add_class_args(parser, cls, ignore_keys=[], group=None):
    group = group or cls.__name__
    ignore_keys = ignore_keys.copy()
    ignore_keys.append("self")
    add_args(parser, cls.__init__, ignore_keys, group)


def get_kwargs(args, callable):
    parameters = inspect.signature(callable).parameters
    kw = {}
    for key in parameters.keys():
        if hasattr(args, key):
            kw[key] = getattr(args, key)
    return kw


def quick_cli(func):

    def wrapper():
        parser = argparse.ArgumentParser()
        add_args(parser, func)
        args = parser.parse_args()
        kw = get_kwargs(args, func)
        return func(**kw)

    return wrapper


def load_confs(paths):
    c = OmegaConf.create({})
    for value in paths:
        c = OmegaConf.merge(c, OmegaConf.load(value))
    return c


def load_structured_confs(dataclasses): 
    conf = None 
    for dataclass in dataclasses: 
        if conf is None: 
            conf = OmegaConf.structured(dataclass)
        else: 
            conf = OmegaConf.merge(conf, OmegaConf.structured(conf))
    return conf


class LoadOmegaConf(Action):
    def __init__(
        self,
        option_strings: Sequence[str],
        dest: str,
        nargs: int | str | None = "+",
        default=[],
        required: bool = False,
        help: str | None = None,
        metavar: str | tuple[str, ...] | None = None,
    ) -> None:
        if not isinstance(default, Sequence): 
            default=[default]
        if len(default) == 0 or isinstance(default[0], str):
            default = load_confs(default)
        elif is_dataclass(default[0]): 
            default = load_structured_confs(default)

        super().__init__(
            option_strings,
            dest,
            nargs=nargs,
            default=default,
            required=required,
            help=help,
            metavar=metavar,
        )

    def __call__(
        self,
        parser: ArgumentParser,
        namespace: Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,
    ) -> None:
        from omegaconf import OmegaConf

        c = load_confs(values)

        setattr(namespace, self.dest, c)


class OverrideYaml(Action):
    def __init__(
        self,
        option_strings: Sequence[str],
        dest: str,
        nargs: int | str | None = "+",
        default=[],
        required: bool = False,
        help: str | None = None,
        metavar: str | tuple[str, ...] | None = "key=value",
    ) -> None:
        super().__init__(
            option_strings,
            dest,
            nargs=nargs,
            default=default,
            required=required,
            help=help,
            metavar=metavar,
        )

    def __call__(
        self,
        parser: ArgumentParser,
        namespace: Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,
    ) -> None:
        assert hasattr(namespace, self.dest), f"Error in parsing arguments"
        c = getattr(namespace, self.dest)
        c = OmegaConf.merge(c, OmegaConf.from_dotlist(values))
        setattr(namespace, self.dest, c)

        parser._defaults[self.dest] = c