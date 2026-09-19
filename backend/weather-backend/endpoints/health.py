from motor.motor_asyncio import AsyncIOMotorDatabase


def get_health(db: AsyncIOMotorDatabase) -> dict:
    return {"api_status": "OK", "db_status": "OK" if db.connect is True else "DB connection not established"}
