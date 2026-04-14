from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import app.model_loader as model_loader
from app.routes import router


@asynccontextmanager
async def lifespan(application: FastAPI):
    model_loader.load()
    yield


app = FastAPI(
    title="Healthcare Test Result Predictor",
    description=(
        "Predict patient test results (Normal / Abnormal / Inconclusive) "
        "from clinical and demographic features. Model is retrained every "
        "Saturday at 12:00 UTC via GitHub Actions."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(router)
