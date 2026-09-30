from pathlib import Path

# Repository structure
EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent

# Input resources
DATA_DIR = REPO_ROOT / "data"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"

# Generated evaluation files
WORK_DIR = EVAL_DIR / "work"
OUTPUT_DIR = EVAL_DIR / "outputs"
PREDICTION_DIR = OUTPUT_DIR / "predictions"

# Create generated-output directories if needed
WORK_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)