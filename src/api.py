"""API module for exposing LoanGuard digital lending model as endpoints."""
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# pylint: disable=import-error
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference import LoanInference

from src.logging_config import get_logger
logger = get_logger(__name__)

# Configuration: environment variable overrides
ENV_MODEL_PATH = os.getenv("LOANGUARD_MODEL_PATH")
ENV_AUTO_SETUP = os.getenv(
    "LOANGUARD_AUTO_SETUP", "0"
)  # set to "1" to allow automatic setup.py run

MODEL_FILENAME = "loanguard_model.joblib"


def candidate_model_paths() -> List[Path]:
    """Return an ordered list of candidate model paths to try."""
    candidates: List[Path] = []

    if ENV_MODEL_PATH:
        candidates.append(Path(ENV_MODEL_PATH))

    candidates.append(Path(__file__).resolve().parents[1] / "models" / MODEL_FILENAME)
    candidates.append(Path.cwd() / "models" / MODEL_FILENAME)
    candidates.append(Path(__file__).resolve().parents[2] / "models" / MODEL_FILENAME)
    candidates.append(Path(__file__).resolve().parents[3] / "models" / MODEL_FILENAME)

    for root in (Path("/mount"), Path("/mnt"), Path("/workspace"), Path("/home")):
        candidates.append(root / MODEL_FILENAME)

    seen = set()
    uniq: List[Path] = []
    for p in candidates:
        try:
            key = str(p.resolve())
        except Exception:  # pylint: disable=broad-exception-caught
            key = str(p)
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def find_model_path() -> Optional[Path]:
    """Return the first existing model path from candidates, or None if not found."""
    for p in candidate_model_paths():
        try:
            if p.exists():
                return p
        except Exception:  # pylint: disable=broad-exception-caught
            continue
    if ENV_MODEL_PATH:
        try:
            return Path(ENV_MODEL_PATH)
        except Exception:  # pylint: disable=broad-exception-caught
            return None
    return None


app = FastAPI(title="LoanGuard Digital Lending API", version="1.0")
predictor: Optional[LoanInference] = None
resolved_model_path: Optional[Path] = None


class LoanInput(BaseModel):
    """Pydantic model representing the schema for loan input data."""
    credit_score: int = Field(..., ge=300, le=850, description="Credit score (300-850)")
    annual_income: int = Field(..., ge=0, description="Gross annual income in USD")
    loan_amount: int = Field(..., ge=100, description="Requested principal in USD")
    dti_ratio: float = Field(..., ge=0.0, le=1.0, description="DTI ratio (0.0-1.0)")
    employment_tenure: int = Field(..., ge=0, description="Employment length (years)")


def try_run_setup() -> Dict[str, Any]:
    """Attempt to run setup.py in the repository root to generate the model artifact."""
    setup_py = Path(__file__).resolve().parents[1] / "setup.py"
    if not setup_py.exists():
        logger.error("setup.py not found at %s", setup_py)
        return {
            "success": False,
            "reason": f"setup.py not found at {setup_py}",
            "stdout": "",
            "stderr": "",
        }

    try:
        logger.info("Running setup.py from %s", setup_py)
        proc = subprocess.run(
            [sys.executable, str(setup_py)], capture_output=True, text=True, check=False
        )
        return {
            "success": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("Failed to run setup.py: %s", exc)
        return {"success": False, "reason": str(exc), "stdout": "", "stderr": ""}


@app.on_event("startup")
def load_model():
    """Load the model during FastAPI startup."""
    # pylint: disable=global-statement
    global predictor, resolved_model_path

    resolved_model_path = find_model_path()
    if resolved_model_path is None or not resolved_model_path.exists():
        if ENV_AUTO_SETUP == "1":
            result = try_run_setup()
            if result.get("success"):
                resolved_model_path = find_model_path()

    if resolved_model_path is None or not resolved_model_path.exists():
        candidates = [str(p) for p in candidate_model_paths()]
        logger.critical("Model artifact not found. Searched: %s", candidates)
        raise RuntimeError(
            f"Model artifact not found. Searched locations: {candidates}.\n"
            "To fix: run 'python setup.py' from repo root to generate the model,\n"
            "or set LOANGUARD_MODEL_PATH to the artifact's absolute path,\n"
            "or set LOANGUARD_AUTO_SETUP=1 to allow the service to run setup.py "
            "automatically (not recommended for production)."
        )

    logger.info("Loading model from %s", resolved_model_path)
    predictor = LoanInference(resolved_model_path)


@app.post("/predict")
def predict_default(loan: LoanInput):
    """Generate default prediction for an applicant."""
    logger.info("Received request for /predict")
    if predictor is None:
        logger.error("Predictor not initialized")
        raise HTTPException(status_code=503, detail="Model core uninitialized")

    payload = loan.model_dump() if hasattr(loan, "model_dump") else loan.dict()
    try:
        decision = predictor.predict(**payload)
        logger.info("Prediction generated successfully")
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("Prediction error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(exc)}") from exc
    return {"input": payload, "decision": decision}


@app.get("/metrics")
def get_metrics():
    """Retrieve performance metrics of the loaded model."""
    if predictor is None:
        return {}
    return predictor.metrics


@app.get("/health")
def health() -> Dict[str, Any]:
    """Return lightweight health info including resolved model path and availability."""
    return {
        "ready": predictor is not None,
        "model_path": (
            str(resolved_model_path) if resolved_model_path is not None else None
        ),
        "model_exists": (
            resolved_model_path.exists() if resolved_model_path is not None else False
        ),
        "metrics_present": bool(predictor.metrics) if predictor is not None else False,
    }


@app.get("/debug/model-candidates")
def model_candidates() -> Dict[str, Any]:
    """Return the candidate paths and whether they exist (debugging endpoint)."""
    rows = []
    for p in candidate_model_paths():
        try:
            exists = p.exists()
        except Exception:  # pylint: disable=broad-exception-caught
            exists = False
        rows.append({"path": str(p), "exists": exists})
    return {
        "candidates": rows,
        "env_model_path": ENV_MODEL_PATH,
        "auto_setup": ENV_AUTO_SETUP,
    }
