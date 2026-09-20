from motor.motor_asyncio import AsyncIOMotorDatabase


async def get_health(db: AsyncIOMotorDatabase) -> dict:
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
