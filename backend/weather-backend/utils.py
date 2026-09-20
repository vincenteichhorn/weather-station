from bson import ObjectId


def oid_to_str(oid):
    """Convert an ObjectId to a string."""
    return str(oid) if isinstance(oid, ObjectId) else oid
