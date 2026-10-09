"""
FastAPI app (ADR-017)
=====================
    uvicorn api.main:app --reload          # from the repo root; http://127.0.0.1:8000/docs

    GET /health                 release being served
    GET /resolve?q=             URL / slug / condition ID / text → markets
    GET /predict?market_id=     every production model's P(YES) as of now, with warnings (ADR-025)

The release is the S3 PRODUCTION pointer unless $RELEASE names one. Lookups are
not stored yet (needs the snapshots migration; ROADMAP step 5).
"""

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api import ROOT, gamma
from api import releases
from api.predict import PredictError, predict
from features import FetchError

load_dotenv(ROOT / ".env")
log = logging.getLogger(__name__)

ERROR_STATUS = {"not_found": 404}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    app.state.release = releases.load()
    yield


app = FastAPI(title="Predicting Prediction Markets", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(","),
    allow_methods=["GET"],
)


def error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": code, "message": message})


@app.get("/health")
def health():
    rel = app.state.release
    return {"status": "ok", "release": rel.name, "dataset_version": rel.dataset_version,
            "models": list(rel.models), "headline_model": rel.headline}


@app.get("/resolve")
def resolve(q: str = Query(..., min_length=1, max_length=500)):
    try:
        markets = gamma.resolve(q)
    except gamma.GammaError as e:
        log.warning("resolve %r: %s", q, e)
        return error(502, "upstream", "Polymarket could not be reached.")
    return {"markets": [gamma.summary(m) for m in markets if m.get("conditionId")]}


@app.get("/predict")
def predict_market(market_id: str = Query(..., pattern=gamma.CONDITION_ID.pattern)):
    try:
        return predict(market_id, app.state.release)
    except PredictError as e:
        return error(ERROR_STATUS.get(e.code, 400), e.code, str(e))
    except (gamma.GammaError, FetchError) as e:
        log.warning("predict %s: %s", market_id, e)
        return error(502, "upstream", "Polymarket could not be reached.")
