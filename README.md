# ManuSense — AI-Powered Manufacturing Quality & Root-Cause Intelligence

> "ManuSense doesn't just detect defects — it understands the manufacturing process behind them."

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-ResNet18-orange)](https://pytorch.org)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-99.11%25-brightgreen)]()

---

## What It Does

ManuSense is an end-to-end surface defect inspection platform. Upload a manufacturing surface image and the system:

1. **Classifies the defect** using a fine-tuned ResNet18 (99.11% held-out accuracy)
2. **Explains the prediction** with Grad-CAM heatmaps showing exactly where the model looked
3. **Assesses reliability** — flags domain shift, low confidence, narrow prediction margin
4. **Generates a prioritised corrective-action plan** (Immediate → Investigate → Correct → Verify)
5. **Links to synthetic process evidence** — station cycle times, bottleneck analysis, defect rates by shift
6. **Estimates cost impact** — scrap, rework, and downtime losses

---

## Defect Classes

| Class | Description |
|-------|-------------|
| `crack` | Linear / branching surface discontinuities |
| `hole` | Circular or irregular material removal |
| `normal` | No defect — uniform surface texture |
| `rust` | Oxidation / corrosion discolouration |
| `scratch` | Linear abrasion from handling or conveyor contact |

---

## Model Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | **99.11%** |
| Weighted F1 | **99.11%** |
| Test Set Size | 1,800 images (15% stratified split) |
| Architecture | ResNet18 fine-tuned from ImageNet |
| Primary failure mode | Rust → Hole (14 / 1,800 errors) |

---

## App Features

- 🔍 **Single Inspection** — upload one image, get full analysis + Grad-CAM + action plan
- 📦 **Batch Upload** — inspect multiple images at once, download CSV summary
- 📋 **Session History** — live log of every inspection this session with confidence trend chart
- 🖼️ **Reference Gallery** — Grad-CAM examples for all 5 classes side-by-side
- 🧪 **What-If Explorer** — simulate any class/confidence combination, explore the action plan
- 📊 **Model Performance** — confusion matrix, per-class metrics, dataset provenance
- 🌙 **Dark Mode** — toggle in the sidebar
- ⚙️ **Adjustable Decision Threshold** — move the ACCEPT/REJECT/REVIEW boundary live

---

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/ManuSense.git
cd ManuSense
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment
```bash
cp .env.example .env
# Edit .env if needed (defaults work out of the box)
```

### 5. Run the app
```bash
streamlit run app/streamlit_app.py
```

Then open **http://localhost:8501** in your browser.

---

## Project Structure

```
ManuSense/
├── app/
│   ├── streamlit_app.py        # Main Streamlit UI (all tabs)
│   ├── config.py               # Settings loaded from .env
│   ├── analytics/              # Bottleneck, metrics, recommendations
│   ├── data/                   # Synthetic data generation
│   ├── economics/              # Profitability modelling
│   ├── recommendations/        # Action plan engine
│   └── vision/
│       ├── predictor.py        # ResNet18 inference wrapper
│       ├── train.py            # Training script
│       ├── evaluate.py         # Evaluation script
│       ├── explainability.py   # Grad-CAM implementation
│       └── preprocessing.py    # Image transforms
├── models/
│   └── best_model.pt           # Trained ResNet18 checkpoint (42 MB)
├── reports/
│   ├── test_metrics.json       # Held-out evaluation results
│   ├── confusion_matrix_test.png
│   ├── gradcam_summary.json
│   └── gradcam/                # Per-class Grad-CAM overlay images
├── data/
│   └── processed/              # Synthetic CSV datasets
├── .env.example                # Environment variable template
├── requirements.txt
└── README.md
```

---

## Training Your Own Model

The trained checkpoint (`models/best_model.pt`) is included — no training required to run the app.

To retrain from scratch (requires the dataset in `train/`):
```bash
python -m app.vision.train
```

To re-evaluate on the held-out test set:
```bash
python -m app.vision.run_eval_only
```

---

## Data Provenance

| Data | Source | Used for |
|------|--------|----------|
| 12,000 inspection images | Challenge dataset | Model training & evaluation |
| Grad-CAM overlays | Generated from trained model | UI explanation |
| Process / production records | **Synthetic** (generated) | Process Evidence tab |
| Economic figures | **Synthetic** (illustrative) | Cost Impact tab |

All synthetic data is clearly labelled in the UI. No real factory production or financial data was available with the challenge dataset.

---

## Requirements

- Python 3.10+
- See `requirements.txt` for full dependency list
- Key packages: `torch`, `torchvision`, `streamlit`, `plotly`, `Pillow`, `pandas`, `numpy`

---

## Notes for Evaluators

- The model weights are included in the repository (`models/best_model.pt`) — the app runs immediately after `pip install -r requirements.txt`
- No GPU required — the model runs on CPU
- The `.env` file is gitignored; copy `.env.example` to `.env` before running
- All monetary and process values shown in the UI are synthetic and clearly labelled as such
