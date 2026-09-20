"""Application entry point and MongoDB lifecycle management."""

from contextlib import asynccontextmanager
import os

from fastapi import FastAPI
from motor.motor_asyncio import AsyncIOMotorClient
from router import router
from dotenv import load_dotenv

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open MongoDB when the application starts and close it on shutdown.

    MongoDB connection settings are read from environment variables. The
    database handle is attached to the FastAPI application for route access.
    """
    mongodb_user = os.getenv("MONDODB_USER", "weather")
    mongodb_password = os.getenv("MONGODB_PASSWORD", "")
    mongodb_host = os.getenv("MONGODB_HOST", "localhost")
    mongodb_port = os.getenv("MONGODB_PORT", "27017")
    mongodb_auth_source = os.getenv("MONGODB_AUTH_SOURCE", "admin")
    app.mongodb_client = AsyncIOMotorClient(f"mongodb://{mongodb_user}:{mongodb_password}@{mongodb_host}:{mongodb_port}?authSource={mongodb_auth_source }")
    app.mongodb = app.mongodb_client[os.getenv("MONGODB_DB", "weather_db")]
    print("Connected to MongoDB")

    yield

    app.mongodb_client.close()
    print("Closed MongoDB connection")


app = FastAPI(title="Weather API", lifespan=lifespan)
app.include_router(router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=int(os.getenv("PORT", 8000)), reload=True)
