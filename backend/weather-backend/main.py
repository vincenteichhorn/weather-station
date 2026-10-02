"""Application entry point and MongoDB lifecycle management."""

from contextlib import asynccontextmanager
import json
import os
from pathlib import Path

from fastapi import FastAPI
from motor.motor_asyncio import AsyncIOMotorClient
import torch
from weather_forecast.model import WeatherProbabilisticTCN
from router import router
from dotenv import load_dotenv

load_dotenv()


def create_mongodb_client():
    """Create a MongoDB client using environment variables."""
    mongodb_user = os.getenv("MONGODB_USER", "weather")
    mongodb_password = os.getenv("MONGODB_PASSWORD")
    mongodb_host = os.getenv("MONGODB_HOST", "localhost")
    mongodb_port = os.getenv("MONGODB_PORT", "27017")
    if not mongodb_password:
        return AsyncIOMotorClient(f"mongodb://{mongodb_host}:{mongodb_port}")

    mongodb_auth_source = os.getenv("MONGODB_AUTH_SOURCE", "admin")
    return AsyncIOMotorClient(
        f"mongodb://{mongodb_user}:{mongodb_password}@{mongodb_host}:{mongodb_port}"
        f"?authSource={mongodb_auth_source}"
    )


def load_forecast_model():
    """Load the forecast model from the specified path."""
    backend_dir = Path(__file__).resolve().parents[1]
    model_path = Path(
        os.getenv(
            "FORECAST_MODEL_PATH",
            "forecast-models/weather-probabilistic-tcn/model.pth",
        )
    )
    config_path = Path(
        os.getenv(
            "FORECAST_MODEL_CONFIG_PATH",
            "forecast-models/weather-probabilistic-tcn/config.json",
        )
    )
    if not model_path.is_absolute():
        model_path = backend_dir / model_path
    if not config_path.is_absolute():
        config_path = backend_dir / config_path
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    model = WeatherProbabilisticTCN(
        lookback=config["lookback"],
        horizon=config["horizon"],
        hidden_channels=config["hidden_channels"],
        kernel_size=config["kernel_size"],
        levels=config["levels"],
        dropout=config["dropout"],
        min_std=torch.tensor(config["min_std"]),
        normalize=True,
        denormalize=True,
        means=torch.tensor(config["normalization"]["means"]),
        stds=torch.tensor(config["normalization"]["stds"]),
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    return model, config


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open MongoDB when the application starts and close it on shutdown.

    MongoDB connection settings are read from environment variables. The
    database handle is attached to the FastAPI application for route access.
    """
    app.mongodb_client = create_mongodb_client()
    app.mongodb = app.mongodb_client[os.getenv("MONGODB_DB", "weather_db")]
    print("Connected to MongoDB")

    app.forecast_model, app.forecast_model_config = load_forecast_model()
    print("Loaded forecast model")

    yield

    app.mongodb_client.close()
    app.forecast_model = None
    app.forecast_model_config = None


app = FastAPI(title="Weather API", lifespan=lifespan)
app.include_router(router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=int(os.getenv("PORT", 8000)), reload=True)
