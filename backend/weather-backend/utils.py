"""Shared conversion helpers used by Pydantic models."""

from bson import ObjectId


def oid_to_str(oid):
    """Convert a MongoDB object identifier to a string when necessary.

    Args:
        oid: Value that may be a :class:`bson.ObjectId`.

    Returns:
        The string representation of an ObjectId, or the original value.
    """
    return str(oid) if isinstance(oid, ObjectId) else oid
