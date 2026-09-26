# Weather Station

## Backend

### Setup

The backend runs a FastAPI server. Dependencies are managed with Poetry. To set up the backend, follow these steps:

```shell
# Install Poetry if you haven't already
cd backend
poetry install
source $(poetry env info --path)/bin/activate

python weather-backend/main.py
```
