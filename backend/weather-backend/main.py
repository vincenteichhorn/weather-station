from contextlib import asynccontextmanager
import os

from fastapi import FastAPI
from motor.motor_asyncio import AsyncIOMotorClient
from router import router
from dotenv import load_dotenv

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.mongodb_client = AsyncIOMotorClient(f"mongodb://{os.getenv('MONGODB_HOST', 'localhost')}:{os.getenv('MONGODB_PORT', '27017')}")
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
