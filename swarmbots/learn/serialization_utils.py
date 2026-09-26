import inspect
from dataclasses import fields, is_dataclass
from enum import Enum
from collections.abc import Callable
from typing import Any


def serialize_fn(fn: Callable) -> dict[str, str]:
    fn_dict = {
        "repr": str(fn),
    }
    try:
        fn_dict["source"] = inspect.getsource(fn)
    except (OSError, TypeError) as err:
        fn_dict["source"] = str(err)
    return fn_dict


def serialize_value(value: Any) -> Any:
    if is_dataclass(value):
        return serialize_dataclass(value)
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, tuple):
        return [serialize_value(item) for item in value]
    if isinstance(value, list):
        return [serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): serialize_value(inner_value) for key, inner_value in value.items()}
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if callable(value):
        if hasattr(value, "__module__") and hasattr(value, "__qualname__"):
            return f"{value.__module__}.{value.__qualname__}"
        if hasattr(value, "__name__"):
            return value.__name__
        return str(value)
    return value


def serialize_dataclass(config: Any) -> dict[str, Any]:
    if not is_dataclass(config):
        raise TypeError(f"Expected dataclass instance, got {type(config)}")
    return {
        field_info.name: serialize_value(getattr(config, field_info.name))
        for field_info in fields(config)
    }
