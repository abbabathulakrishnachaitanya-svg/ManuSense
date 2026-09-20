"""
app/config.py
─────────────
Central configuration loader for ManuSense.

All settings are read from environment variables (populated by .env via
python-dotenv).  Pydantic-Settings validates types and provides defaults,
so the rest of the codebase can import a single `settings` object without
touching os.environ directly.
"""

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────────────
    app_env: str = Field("development", description="development | production")
    log_level: str = Field("INFO", description="DEBUG | INFO | WARNING | ERROR")

    # ── Data paths ───────────────────────────────────────────────────────────
    data_dir: Path = Field(Path("data"), description="Root data directory")
    images_dir: Path = Field(Path("data/images"), description="Raw inspection images")
    processed_dir: Path = Field(Path("data/processed"), description="Pre-processed artefacts")
    models_dir: Path = Field(Path("models"), description="Saved model weights")

    # Dataset root — organizer images live in train/ at the project root
    train_dir: Path = Field(Path("train"), description="ImageFolder-compatible dataset root")

    # ── Vision model ─────────────────────────────────────────────────────────
    model_name: str = Field(
        "resnet18",
        description="Architecture: resnet18 | mobilenet_v3_small",
    )
    model_weights_path: Path = Field(
        Path("models/best_model.pt"),
        description="Path to saved best-model checkpoint",
    )
    image_size: int = Field(224, description="Square input resolution fed to the model")
    num_classes: int = Field(5, description="Number of output classes")

    # ── Training hyperparameters ──────────────────────────────────────────────
    train_val_split: float = Field(0.8, ge=0.0, le=1.0, description="Fraction used for training")
    random_seed: int = Field(42, description="Seed for reproducible splits and weight init")
    batch_size: int = Field(32, gt=0)
    num_epochs: int = Field(15, gt=0)
    learning_rate: float = Field(0.001, gt=0.0)
    # num_workers > 0 speeds up loading on Linux/macOS; keep 0 on Windows to
    # avoid multiprocessing spawn issues with PyTorch DataLoader.
    num_workers: int = Field(0, ge=0)

    # ── Decision thresholds ───────────────────────────────────────────────────
    # These are starting points — not claimed to be optimal.
    # Tune them based on the validation metrics after training.
    #
    # Logic (applied in predictor.py):
    #   confidence >= confidence_threshold  → ACCEPT  (if predicted "normal")
    #                                       → REJECT  (if predicted defect class)
    #   confidence <  uncertainty_threshold → REVIEW  (model is not confident)
    #   otherwise                           → REVIEW  (confidence is between the two thresholds)
    confidence_threshold: float = Field(0.80, ge=0.0, le=1.0)
    uncertainty_threshold: float = Field(0.50, ge=0.0, le=1.0)

    # ── Economics ────────────────────────────────────────────────────────────
    currency: str = Field("USD")
    cost_per_defect: float = Field(0.0, ge=0.0)
    revenue_per_unit: float = Field(0.0, ge=0.0)

    # ── Derived helpers ──────────────────────────────────────────────────────
    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"

    @property
    def defect_classes(self) -> list[str]:
        """All classes that are NOT 'normal'."""
        # This list is derived at runtime from the dataset folders; the
        # property is a convenience alias used by the decision logic.
        return ["crack", "hole", "rust", "scratch"]


# ── Singleton ────────────────────────────────────────────────────────────────
# Import this object everywhere:  from app.config import settings
settings = Settings()
