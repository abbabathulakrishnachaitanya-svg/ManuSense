# ManuSense — Jury Evaluation Document
## AI-Powered Manufacturing Quality & Root-Cause Intelligence

---

## 1. PROBLEM STATEMENT

Manufacturing defect inspection is traditionally done manually — slow, inconsistent, and expensive.
A single missed defect can cause:
- Product recalls costing millions
- Line stoppages and unplanned downtime
- Customer complaints and warranty claims
- Safety risks in critical industries

**ManuSense solves this by automating surface defect detection using computer vision,
and going one step further — explaining WHY a defect happened and WHAT to do about it.**

---

## 2. SOLUTION OVERVIEW

ManuSense is an end-to-end AI manufacturing inspection platform that:

1. Classifies surface defects in real-time using a fine-tuned ResNet18 deep learning model
2. Explains every prediction using Grad-CAM visual heatmaps
3. Assesses prediction reliability (flags uncertain or out-of-distribution images)
4. Generates a 4-phase corrective action plan tied to the specific defect type
5. Links defects to process variables (vibration, pressure, humidity)
6. Estimates the financial cost of each defect decision
7. Supports batch inspection of multiple images at once
8. Maintains a live session history with trend analysis

---

## 3. TECHNICAL FEASIBILITY

### 3.1 Model Performance
| Metric              | Value                              |
|---------------------|------------------------------------|
| Test Accuracy       | 99.11% (1,784 / 1,800 correct)     |
| Weighted F1 Score   | 99.11%                             |
| Test Set Size       | 1,800 images (held-out, unseen)    |
| Training Set        | 8,400 images                       |
| Validation Set      | 1,800 images                       |
| Total Dataset       | 12,000 images, 5 classes           |
| Architecture        | ResNet18, fine-tuned from ImageNet |
| Training Strategy   | 70/15/15 stratified split, seed 42 |
| Primary Error Mode  | Rust → Hole (14/1,800 = 0.78%)     |

### 3.2 Technology Stack
| Component         | Technology                        |
|-------------------|-----------------------------------|
| Deep Learning     | PyTorch + torchvision             |
| Model             | ResNet18 (pretrained, fine-tuned) |
| Explainability    | Grad-CAM (custom implementation)  |
| UI Framework      | Streamlit                         |
| Charts            | Plotly                            |
| Data Processing   | Pandas, NumPy                     |
| Image Processing  | Pillow (PIL)                      |
| Runtime           | CPU (no GPU required)             |

### 3.3 Why ResNet18?
- Proven architecture for image classification tasks
- Fast inference on CPU — suitable for edge deployment
- Small enough to ship with the repository (42 MB checkpoint)
- Fine-tuning from ImageNet weights gives excellent results even with limited data
- Grad-CAM is naturally compatible with CNN architectures

### 3.4 Reliability System
ManuSense does not blindly trust the model. It checks:
- **Confidence threshold** — predictions below threshold are flagged for manual review
- **Prediction margin** — if top-1 and top-2 probabilities are close (< 30% gap), flags as uncertain
- **Domain shift detection** — checks image brightness and contrast against training distribution
- **Known confusion pattern** — warns when rust/hole confusion is likely
- **Three-state decision** — ACCEPT / REJECT / REVIEW (not binary)

---

## 4. ECONOMIC FEASIBILITY

### 4.1 Cost of the Solution
| Item                          | Cost                              |
|-------------------------------|-----------------------------------|
| Development                   | Open-source stack, zero licensing |
| Deployment                    | Runs on standard CPU hardware     |
| GPU                           | Not required                      |
| Cloud                         | Optional (can run fully on-premise)|
| Model retraining              | One-time per product line         |

### 4.2 Value Delivered
| Benefit                          | Impact                              |
|----------------------------------|-------------------------------------|
| Inspection speed                 | Real-time vs. minutes manually      |
| Consistency                      | 99.11% accuracy vs. human variance  |
| Early defect detection           | Prevents downstream scrap buildup   |
| Root-cause guidance              | Reduces engineer investigation time |
| Batch processing                 | 100s of images in minutes           |
| Exportable reports               | Audit trail, no manual logging      |

### 4.3 Illustrative Cost Savings (Synthetic Model)
Using conservative industry assumptions:
- Scrap cost per defective unit: $25
- Rework cost per reworked unit: $10
- Downtime cost: $2.50 per minute
- At a 5% defect rate on 500 units/day:
  - **Daily scrap savings potential: ~$625**
  - **Annual savings potential: ~$228,000**

*Note: All monetary figures are illustrative. Real savings depend on actual factory data.*

---

## 5. OPERATIONAL FEASIBILITY

### 5.1 Ease of Use
- Browser-based UI — no installation needed for end users
- Single image upload → full report in seconds
- Batch upload for high-volume inspection lines
- Dark mode for factory floor visibility
- Adjustable decision threshold — operators can tune sensitivity
- Downloadable CSV reports for quality records

### 5.2 Integration Path
| Stage        | What Happens                                           |
|--------------|--------------------------------------------------------|
| MVP (now)    | Standalone web app, manual image upload                |
| Phase 2      | REST API wrapper around predictor.py for line integration |
| Phase 3      | Camera feed integration, real-time inference pipeline  |
| Phase 4      | MES/ERP integration via Inspection ID linkage          |

### 5.3 Scalability
- Stateless inference — each prediction is independent
- Can be containerised (Docker) for cloud or edge deployment
- Horizontal scaling: multiple instances behind a load balancer
- Model retraining pipeline already implemented (app/vision/train.py)

---

## 6. SOCIAL / ENVIRONMENTAL FEASIBILITY

### 6.1 Social Impact
- Reduces repetitive, eye-strain inducing manual inspection work
- Upskills operators from visual checkers to exception handlers
- Improves product safety — fewer defective products reach consumers
- Transparent AI — every decision explained via Grad-CAM

### 6.2 Environmental Impact
- Reduces material waste by catching defects earlier in the process
- Less scrap = less raw material consumption
- Less rework = less energy used re-processing defective units
- CPU-only inference = low energy footprint vs. GPU-heavy alternatives

---

## 7. LEGAL / ETHICAL FEASIBILITY

### 7.1 Data Privacy
- No personal data is collected or processed
- Inspection images are processed in-memory, not stored
- No cloud transmission of proprietary product images

### 7.2 AI Ethics
- Model decisions are ADVISORY only — no automatic machine control
- Every prediction includes a reliability flag and confidence score
- Human-in-the-loop: REVIEW state routes uncertain predictions to human inspectors
- Grad-CAM provides full transparency into model reasoning
- Known failure modes are documented and warned about in the UI

### 7.3 Intellectual Property
- Built entirely on open-source frameworks (MIT/Apache licensed)
- Training data: challenge-provided dataset
- No proprietary third-party models or APIs used

---

## 8. COMPETITIVE ADVANTAGE

| Feature                        | ManuSense | Generic Classifier | Manual Inspection |
|--------------------------------|-----------|--------------------|-------------------|
| 99%+ accuracy                  | ✅        | Varies             | ~85-95%           |
| Grad-CAM explanation           | ✅        | ❌                 | N/A               |
| 4-phase corrective action plan | ✅        | ❌                 | Manual            |
| Reliability / uncertainty flag | ✅        | ❌                 | N/A               |
| Batch processing               | ✅        | Varies             | Slow              |
| Session history & export       | ✅        | ❌                 | Paper-based       |
| Process variable linkage       | ✅        | ❌                 | N/A               |
| Cost impact estimation         | ✅        | ❌                 | N/A               |
| No GPU required                | ✅        | Often ❌           | N/A               |
| Runs on-premise                | ✅        | Sometimes ❌       | N/A               |

---

## 9. LIMITATIONS & HONEST DISCLOSURES

| Limitation                          | Mitigation                                     |
|-------------------------------------|------------------------------------------------|
| Trained on specific challenge images | Domain shift detection warns when image differs|
| No bounding-box localisation        | Grad-CAM gives approximate region highlight    |
| Process data is synthetic           | Clearly labelled in UI; ready for real data    |
| Economics figures are illustrative  | Clearly labelled; configurable via .env        |
| Rust→Hole confusion (0.78% of test) | Confusion warning shown when likely            |
| No real-time camera feed (MVP)      | REST API integration planned for Phase 2       |

---

## 10. QUICK START FOR JURY

```bash
# 1. Clone
git clone https://github.com/abbabathulakrishnachaitanya-svg/ManuSense.git
cd ManuSense

# 2. Install
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# 3. Configure
cp .env.example .env

# 4. Run
streamlit run app/streamlit_app.py
# Open http://localhost:8501
```

**The trained model is included** — no training required. Upload any image from the
`reports/gradcam/` folder to immediately see the full pipeline in action.

---

## 11. REPOSITORY

**GitHub:** https://github.com/abbabathulakrishnachaitanya-svg/ManuSense

**Key files:**
- `app/streamlit_app.py` — Full UI (2000+ lines, all features)
- `app/vision/predictor.py` — ResNet18 inference wrapper
- `app/vision/explainability.py` — Grad-CAM implementation
- `app/vision/train.py` — Training pipeline
- `models/best_model.pt` — Trained checkpoint (42 MB)
- `reports/test_metrics.json` — Held-out evaluation results

---

*ManuSense — Built for the manufacturing floor. Transparent, explainable, actionable.*
