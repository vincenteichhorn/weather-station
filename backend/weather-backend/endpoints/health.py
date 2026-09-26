"""Health check operations for the API."""

from motor.motor_asyncio import AsyncIOMotorDatabase


async def get_health(db: AsyncIOMotorDatabase) -> dict:
    """Check whether MongoDB responds to a ping command.

    Args:
        db: Database handle used for the connectivity check.

    Returns:
        A mapping containing API status, database status, and an optional
        exception message.
    """
    try:
        await db.command("ping")
        return {
            "api_status": "OK",
            "db_status": "OK",
            "exception": None,
        }
    except Exception as exception:
        return {
            "api_status": "OK",
            "db_status": "DB connection not established",
            "exception": str(exception),
        }
