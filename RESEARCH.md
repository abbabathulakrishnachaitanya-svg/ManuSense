# ManuSense — Research & Technical Approach

## 1. Problem Understanding

### Problem Statement

**Visual Inspection & Defect Root-Cause Assistant**

Manufacturing lines can produce large volumes of products where subtle defects, process variation, bottlenecks, work-in-progress, downtime, changeovers, mixed product variants and changing operating conditions affect quality and profitability.

A useful software solution therefore needs to go beyond a simple image classifier.

ManuSense is designed as a decision-support pipeline that connects:

```
IMAGE → DEFECT → PROCESS EVIDENCE → CONSTRAINT → COST → DECISION
```

The system has four major analytical layers:

1. Visual quality inspection
2. Process and root-cause analytics
3. Production-flow and bottleneck analytics
4. Economic and profitability simulation

The final layer produces evidence-based advisory recommendations.

---

## 2. Data Provenance

### 2.1 Organizer-provided inspection data

The visual inspection images are from the dataset provided for the challenge.
We use those images for the actual defect classification component.

| Class | Images |
|-------|-------:|
| normal | 2,400 |
| crack | 2,400 |
| hole | 2,400 |
| rust | 2,400 |
| scratch | 2,400 |
| **Total** | **12,000** |

- Format: grayscale PNG, 256×256
- Location: `train/` at the project root (ImageFolder-compatible layout)

### 2.2 Missing production and economic data

The challenge did not provide production-flow and economic datasets along with the image dataset.
So for demonstrating the complete architecture, we created clearly labelled synthetic production and economics data.
We do not present those values as real factory measurements.
They are used only to demonstrate root-cause analysis, bottleneck analysis and profitability simulation.

### 2.3 Synthetic data fields (planned)

| Field | Purpose |
|-------|---------|
| `unit_id` | Links inspection record to production record |
| `image_path` | Links to inspection image |
| `batch_id` | Batch-level analysis |
| `station_id` | Station comparison |
| `product_variant` | Variant analysis |
| `shift` | Shift analysis |
| `timestamp` | Time and trend analysis |
| `cycle_time_seconds` | Flow performance |
| `downtime_minutes` | Availability |
| `changeover_minutes` | Changeover impact |
| `wip` | Work in progress |
| `temperature` | Process condition |
| `pressure` | Process condition |
| `machine_speed` | Process condition |
| `vibration` | Process condition |
| `humidity` | Process condition |

Controlled synthetic associations are created so the analytics demonstrate meaningful patterns.
These are simulation assumptions, not real industrial findings.

Examples of synthetic associations used:
- vibration associated with crack and scratch rates
- pressure associated with hole rates
- temperature and humidity associated with rust rates
- certain stations having higher defect rates
- changeover periods associated with temporary quality changes
- high WIP around capacity-constrained stations

---

## 3. System Architecture

```
                    INSPECTION DATA (real)
                           │
                           ▼
                  ┌─────────────────┐
                  │ Image Processing│
                  └────────┬────────┘
                           ▼
                  ┌─────────────────┐
                  │  ResNet18       │
                  │  Classifier     │
                  └────────┬────────┘
                           │
               ┌───────────┼───────────┐
               ▼           ▼           ▼
            Defect     Confidence   Grad-CAM
                           │
                           ▼
                    Unified Records
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
       Root-Cause Analytics       Flow Analytics
       (synthetic data)           (synthetic data)
              │                         │
              └────────────┬────────────┘
                           ▼
                   Economic Simulation
                   (synthetic data)
                           │
                           ▼
                    Recommendations
                           │
                           ▼
                      Dashboard
```

---

## 4. Vision Inspection Approach

### 4.1 Architecture choice — ResNet18

ResNet18 was chosen because:
- It is lightweight enough for CPU-only hackathon hardware (~11 M parameters)
- ImageNet-pretrained weights are publicly available via `torchvision`
- The final fully-connected layer can be replaced with a 5-class head in two lines
- Training converges quickly on a 12,000-image dataset

MobileNetV3-Small is an alternative if inference speed is the priority.

### 4.2 Grayscale adaptation

ResNet18 expects 3-channel RGB input.
The inspection images are grayscale (single channel).

**Approach:** `transforms.Grayscale(num_output_channels=3)` repeats the single channel three times before passing it to the network.
This preserves all pixel information and allows the pretrained conv1 weights to be reused without modification.
The alternative (changing conv1 in_channels to 1) would discard pretrained weights for the first layer.

### 4.3 Transfer learning strategy

All layers are fine-tuned rather than freezing the backbone.
With 12,000 images and a 24-hour time constraint, full fine-tuning on a small ResNet18 converges quickly.
The domain (industrial texture images) is visually dissimilar from ImageNet photos, so fine-tuning the full network typically outperforms a frozen backbone.

### 4.4 Train/validation split

- Split ratio: configurable via `TRAIN_VAL_SPLIT` (default 0.80)
- Method: `torch.utils.data.random_split` with a fixed `Generator` seed (`RANDOM_SEED=42`)
- This gives the same indices every run, which is required for honest evaluation

### 4.5 Transform pipeline

Training transforms:
- `Grayscale(num_output_channels=3)`
- `Resize(224, 224)`
- `RandomHorizontalFlip()`
- `RandomRotation(degrees=10)`
- `ToTensor()`
- `Normalize(ImageNet mean, ImageNet std)`

Validation transforms (no augmentation):
- `Grayscale(num_output_channels=3)`
- `Resize(224, 224)`
- `ToTensor()`
- `Normalize(ImageNet mean, ImageNet std)`

ImageNet normalisation is used because we start from ImageNet-pretrained weights.

Augmentation is intentionally conservative.
Aggressive augmentation (colour jitter, large crops) would distort industrial defect morphology.

### 4.6 The `_TransformSubset` pattern

`torch.utils.data.random_split` returns Subset objects that share the same underlying `ImageFolder`.
If `ImageFolder` is given its own transform, both train and val subsets receive the same transforms — including training augmentation applied to the validation set.

`_TransformSubset` wraps each Subset and applies the correct per-split transform at `__getitem__` time, solving this problem cleanly without duplicating image data.

---

## 5. Confidence-Aware Decision Policy

Three advisory decision states:

| State | Condition |
|-------|-----------|
| **ACCEPT** | `confidence ≥ CONFIDENCE_THRESHOLD` and predicted class = `normal` |
| **REJECT** | `confidence ≥ CONFIDENCE_THRESHOLD` and predicted class ≠ `normal` |
| **REVIEW** | `confidence < UNCERTAINTY_THRESHOLD` (regardless of class) |
| **REVIEW** | Between the two thresholds |

Thresholds are configurable via `.env`:
- `CONFIDENCE_THRESHOLD` (default 0.80)
- `UNCERTAINTY_THRESHOLD` (default 0.50)

These are initial prototype values, not claims of optimality.
They should be calibrated using the validation confusion matrix and the operational cost of false accepts versus false rejects.

---

## 6. False Accept and False Reject

**False Accept:** a genuinely defective unit is classified as acceptable.
Risk: defective products reach the customer.

**False Reject:** a genuinely acceptable unit is classified as defective.
Risk: unnecessary rework, scrap or manual inspection cost.

The relative cost of these errors is application-specific.
The REVIEW state provides a buffer — uncertain cases go to a human rather than being forced into a binary outcome.

Reported metrics:
- false accept rate (recall of defect classes)
- false reject rate (1 − precision of normal class)
- review/uncertainty rate
- confusion matrix
- per-class precision, recall and F1

---

## 7. Localization and Explainability

The available image dataset does not include bounding-box or segmentation annotations.

ManuSense does **not** claim ground-truth defect localization from these images.

Where supported, Grad-CAM generates a heatmap showing image regions that contributed most to the prediction.
The target layer for ResNet18 is `model.layer4[-1]` — the last convolutional block before global average pooling.

**Model explanation ≠ ground-truth localization.**

If localization annotations become available later, the system can calculate proper localization metrics such as IoU.

---

## 8. Final Vision MVP Evaluation Results

These are the actual results from the held-out test set using checkpoint `models/best_model.pt` (epoch 1, val_acc = 0.9911).
No retraining was performed after evaluation.
All numbers are sourced from `reports/test_metrics.json`.

### Split

| Set | Images | Method |
|-----|-------:|--------|
| Train | 8,400 | Stratified 70 %, seed = 42 |
| Validation | 1,800 | Stratified 15 %, seed = 42 |
| Test | 1,800 | Stratified 15 %, seed = 42 |

### Test set metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | **99.11%** |
| Correct / Incorrect | **1,784 / 16** out of 1,800 |
| Weighted F1 | 99.11% |
| Macro F1 | 99.11% |
| Weighted Precision | 99.14% |
| Weighted Recall | 99.11% |

### Per-class results

| Class | Precision | Recall | F1 | Support |
|-------|----------:|-------:|---:|--------:|
| crack | 1.00 | 1.00 | 1.00 | 360 |
| hole | 0.96 | 1.00 | 0.98 | 360 |
| normal | 0.99 | 1.00 | 1.00 | 360 |
| rust | 1.00 | 0.96 | 0.98 | 360 |
| scratch | 1.00 | 1.00 | 1.00 | 360 |

### Error analysis

All 16 misclassifications were **rust** samples:
- 14 rust images predicted as **hole**
- 2 rust images predicted as **normal**
- crack, hole, normal and scratch all achieved perfect scores on the test set

Rust images with surface pitting can visually resemble hole defects at the texture level. This is a known challenge for single-epoch training on visually similar defect categories.

### Grad-CAM

Grad-CAM overlays were generated for one image per class and saved to `reports/gradcam/`.
Grad-CAM is a **model explanation** showing which image regions influenced the prediction.
It is **not** ground-truth defect localization.
The dataset contains no bounding-box or segmentation annotations.

### Artefacts

- `reports/test_metrics.json` — full metrics in machine-readable form
- `reports/confusion_matrix_test.png` — 5×5 confusion matrix
- `reports/gradcam/gradcam_<class>.png` — one Grad-CAM overlay per class

---

## 9. Robustness to Unseen Conditions

Manufacturing inspection conditions can change due to illumination, contrast, blur, noise, orientation and surface appearance.

A practical prototype can evaluate controlled perturbation test sets:
- brightness changes
- contrast changes
- Gaussian noise
- blur
- small rotations

Measuring performance change between clean and perturbed data gives a practical robustness estimate without requiring additional labeled data.

---

## 10. Novel and Uncertain Defect Handling

The supplied dataset contains five known classes and does not provide labelled unseen-defect examples.

ManuSense does **not** claim validated novel-defect detection from this dataset.

A practical prototype can combine:
- low prediction confidence (below `UNCERTAINTY_THRESHOLD`)
- optional: embedding distance or novelty score

to flag suspicious cases for human review rather than forcing a prediction.

---

## 11. Root-Cause Correlation Engine

Calculates defect statistics across candidate process factors.

Useful outputs:
- defect count per group
- total inspected per group
- defect rate per group
- baseline defect rate (overall)
- relative risk / defect-rate ratio
- sample size
- statistical evidence where appropriate

For continuous process variables, compare distributions or averages between normal and defective observations.

Methods:
- grouped descriptive statistics
- chi-square or Fisher tests for categorical relationships
- t-test or Mann-Whitney U for continuous variables
- logistic regression for exploratory association

### Interpretation rule

Statistical association is not automatically proof of physical causation.

Wording used in output:
- "associated with"
- "higher observed defect rate"
- "candidate contributing factor"

These phrases are used deliberately to avoid overclaiming.

---

## 12. Production Flow and Bottleneck Analysis

Key variables:
- cycle time (seconds per unit)
- scheduled production time
- downtime (minutes)
- changeover time (minutes)
- WIP (units in queue)
- throughput (units per hour)
- effective capacity
- utilization

Simplified capacity calculation:

```
Effective available time = Scheduled time − downtime − changeover time
Estimated capacity = Effective available time / average cycle time
```

For a serial line, the station with the lowest effective capacity is the candidate bottleneck.

When production data is synthetic, output is labelled as **simulated candidate constraint**, not a real factory bottleneck.

---

## 13. Profitability Model

Simplified contribution model:

```
Estimated contribution =
  Revenue
  − material cost
  − processing cost
  − scrap loss (scrap_units × scrap_cost_per_unit)
  − rework loss (rework_units × rework_cost_per_unit)
  − downtime loss (downtime_minutes × cost_per_minute)
```

Inputs come from `.env` (`COST_PER_DEFECT`, `REVENUE_PER_UNIT`) and synthetic economics data.

These are simulations, not financial forecasts.
The profitability module demonstrates the decision-support logic.
It does not report actual factory profitability.

---

## 14. Recommendation Engine

Each recommendation connects:

```
Defect pattern
      ↓
Associated station / process condition (from root-cause analytics)
      ↓
Flow constraint (from bottleneck analytics)
      ↓
Estimated quality / throughput impact
      ↓
Estimated cost impact (from profitability simulation)
      ↓
Advisory recommended action
```

Each recommendation includes:
1. Observation
2. Supporting evidence (association score, defect rate delta)
3. Affected metric
4. Simulated impact estimate
5. Suggested advisory action
6. Confidence level and limitations

Recommendations are advisory only. No machine or process control is performed.

---

## 15. Technology Stack

| Layer | Library | Version |
|-------|---------|---------|
| Vision / ML | PyTorch | 2.3.0 |
| Vision / ML | Torchvision | 0.18.0 |
| Explainability | pytorch-grad-cam | 1.5.2 |
| Data handling | NumPy | 1.26.4 |
| Data handling | Pandas | 2.2.2 |
| Statistics | SciPy | 1.13.0 |
| Metrics | scikit-learn | 1.5.0 |
| Image I/O | Pillow | 10.3.0 |
| Image I/O | OpenCV | 4.9.0 |
| Dashboard | Streamlit | 1.35.0 |
| Charts | Plotly | 5.22.0 |
| Config | Pydantic-Settings | 2.3.0 |
| Config | python-dotenv | 1.0.1 |
| Testing | pytest | 8.2.2 |

---

## 16. Why This Architecture?

A single image classifier addresses only the visual portion of the manufacturing problem.

The proposed architecture separates the questions:

```
Vision
→ What defect is visible?

Root-cause analytics
→ What process factors are associated with the defect?

Flow analytics
→ Where is the production constraint?

Economic simulation
→ What could the quality or flow issue cost?

Recommendation
→ What should the team investigate or simulate next?
```

This provides a traceable path from inspection evidence to operational and economic analysis — which is closer to the actual question a manufacturing quality team needs to answer.

---

## 17. Evaluation Alignment

| Evaluation area | Implementation |
|-----------------|----------------|
| Defect detection / classification accuracy | Accuracy, precision, recall, F1, confusion matrix |
| Localization | Grad-CAM heatmaps; no ground-truth localization claim without annotations |
| Robustness to unseen conditions | Controlled perturbation test protocol |
| False reject / false accept | Separate rates plus REVIEW state |
| Root-cause correlation | Defect-rate and process-factor analysis (synthetic data) |
| Explainability / confidence | Softmax probabilities, configurable thresholds, Grad-CAM |
| Technical implementation | Modular Python / AI / data architecture |
| UI/UX / visualization | Streamlit + Plotly decision dashboard |

---

## 18. MVP Priorities

### P0 — Must work

- Dataset ingestion from `train/`
- Five-class image classification
- Confidence-aware ACCEPT / REJECT / REVIEW decision
- Evaluation metrics (accuracy, precision, recall, F1, confusion matrix)
- Best-model checkpoint saved to `models/best_model.pt`
- Streamlit inspection flow (upload → predict → decision → probabilities)

### P1 — Important

- Grad-CAM explanation overlay
- Synthetic production data generation
- Root-cause analytics
- Bottleneck calculation
- Profitability simulation
- Recommendation engine

### P2 — If time permits

- Novelty / uncertainty scoring
- Calibrated confidence (temperature scaling or Platt scaling)
- Advanced localization (requires annotations)
- Richer statistical analysis
- Robustness perturbation testing

---

## 19. Limitations

1. The available inspection dataset does not contain bounding-box or segmentation annotations. True localization performance cannot be measured from the available labels.
2. Production-flow data is synthetic. Synthetic process associations are not validated industrial causal relationships.
3. Economic data is synthetic. Profitability results are simulations, not financial forecasts.
4. Confidence thresholds (0.80 and 0.50) are initial prototype values and require calibration before any real deployment.
5. Novel-defect detection is not validated without representative unseen-defect labeled data.
6. Recommendations are advisory and simulated. The system does not control production hardware.
7. `NUM_WORKERS=0` is required on Windows due to PyTorch DataLoader multiprocessing constraints. Set higher on Linux/macOS for faster training.

---

## 20. Design Principle

**Detect accurately. Explain clearly. Connect evidence. Quantify impact. Recommend carefully.**
