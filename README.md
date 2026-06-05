<div align="center">

<!-- HEADER BANNER -->
<picture>
  <img src="docs/images/banner_light.png" alt="CoheRex-Integrity" width="100%"/>
</picture>

# CoheRex-Integrity

**Multi-Agent Temporal Consistency Verification & Video Forensics Framework**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-0d1117?style=flat-square&logo=python&logoColor=4fc3f7&labelColor=0d1117&color=4fc3f7)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-0d1117?style=flat-square&logo=ultralytics&logoColor=00e676&labelColor=0d1117&color=00e676)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-0d1117?style=flat-square&logo=streamlit&logoColor=ff4b4b&labelColor=0d1117&color=ff4b4b)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-0d1117?style=flat-square&logo=scikitlearn&logoColor=f9a825&labelColor=0d1117&color=f9a825)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-0d1117?style=flat-square&logoColor=white&labelColor=0d1117&color=6e40c9)](https://opensource.org/licenses/MIT)

<br/>

<img src="docs/images/portal_analyzing.png" alt="CoheRex Portal" width="88%" style="border-radius:12px; box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.4);"/>

*The CoheRex Streamlit Portal processing a video for temporal integrity violations.*

</div>

<br/>

> [!IMPORTANT]
> **CoheRex-Integrity** is a physics-first video forensics engine that detects synthetic temporal tampering — frame deletions, duplications, speed manipulation, clip splicing, and reverse playback — by modeling physical trajectories and human behavior over time rather than relying on pixel-level artifacts. A resilient, architecture-driven approach to video integrity verification.

<br/>

---

## 📌 Table of Contents

1. [What CoheRex Detects](#-what-coherex-detects)
2. [System Architecture](#-system-architecture)
3. [Core Innovations](#-core-innovations)
4. [Tech Stack & Requirements](#-tech-stack--requirements)
5. [Codebase Structure](#-codebase-structure)
6. [Dashboard Portal](#-dashboard-portal)
7. [Setup & Execution](#-setup--execution)
8. [Design Philosophy](#-design-philosophy)

---

## 🔍 What CoheRex Detects

> [!NOTE]
> CoheRex doesn't look at what a frame *looks like* — it looks at whether the **physics of the scene make sense across time**. It flags anomalies at the **trajectory domain**, not the pixel domain.

| Tampering Type | Detection Mechanism |
|:---:|:---|
| 🔴 **Frame Deletion** | Velocity / acceleration discontinuities in tracked trajectories |
| 🟠 **Frame Duplication** | Near-zero MCV z-scores sustained over abnormal windows |
| 🟡 **Speed Manipulation** | Kinematic anomalies inconsistent with natural motion physics |
| 🟣 **Clip Splicing** | Track lifecycle reattachments and identity breaks at cut points |
| 🔵 **Reverse Playback** | Bidirectional trajectory reversal patterns in the motion agent |

---

## 🏛️ System Architecture

The pipeline traverses **six discrete stages** — from raw video frames to a signed forensic verdict.

```mermaid
flowchart TD
    A(["🎬  RAW VIDEO INPUT"]):::start

    subgraph S1["  I · DATA ACQUISITION  "]
        B["Frame Decode\nFPS Extraction"]
    end

    subgraph S2["  II · DETECTION LAYER  "]
        C["YOLOv8 Detector\nBounding Boxes & Classes"]
        D["YOLOv8-Pose Estimator\nKeypoint Geometry"]
    end

    subgraph S3["  III · STATE ESTIMATION  "]
        E["Track Manager\nKalman Filter State Propagation"]
        F["Trajectory Store\nKinematic History per ID"]
    end

    subgraph S4["  IV · MOTION COHERENCE ENGINE  "]
        G["MCV · Rolling z-score Computation\nDynamic Noise Floor Application"]
    end

    subgraph S5["  V · MULTI-AGENT FUSION  "]
        H1["⚡ Motion Agent\nVelocity · Acceleration · Angular Δ"]
        H2["🔗 Continuity Agent\nTrack Lifecycle · Reattachments"]
        H3["👥 Crowd Agent\nDirectional Alignment · Consensus"]
        K["Reliability Estimator\nDynamic Per-Frame Weight Scaling"]
        J["Fusion Engine\nWeighted Score Aggregation"]
    end

    subgraph S6["  VI · CLASSIFICATION & VERDICT  "]
        L["Feature Extractor\nGlobal Temporal Feature Vector"]
        M["Random Forest Classifier\n5-Fold Stratified Cross-Validation"]
        N(["✅  FORENSIC VERDICT"]):::verdict
    end

    subgraph S7["  VII · PRESENTATION LAYER  "]
        O["Streamlit Dashboard\nDual Sync Player · Charts · Reports"]
    end

    A --> B
    B --> C & D
    C --> E
    D --> E
    E --> F
    F --> G
    G --> H1 & H2 & H3
    K --> J
    H1 --> J
    H2 --> J
    H3 --> J
    J --> L
    L --> M
    M --> N
    N --> O

    classDef start fill:#1a1a2e,stroke:#4fc3f7,stroke-width:2px,color:#4fc3f7,font-weight:bold
    classDef verdict fill:#1a1a2e,stroke:#00e676,stroke-width:2px,color:#00e676,font-weight:bold
    classDef default fill:#0d1117,stroke:#30363d,stroke-width:1px,color:#c9d1d9
```

---

## ⚙️ Core Innovations

### ① Multi-Agent Decoupled Architecture

> [!TIP]
> Each agent is **fully independent** — it can be validated, extended, or swapped without affecting the rest of the pipeline. The fusion layer gracefully absorbs whatever signal is trustworthy.

```mermaid
flowchart LR
    IN(["Frame Signal Input"]):::input

    subgraph Agents["  Independent Agent Ensemble  "]
        direction TB
        A1["⚡ Motion Agent
        · Velocity magnitude
        · Acceleration magnitude
        · Angular velocity (heading Δ)
        · MCV z-score"]:::agent

        A2["🔗 Continuity Agent
        · Track reattachment events
        · Dormant frame windows
        · Identity lifecycle breaks"]:::agent

        A3["👥 Crowd Agent
        · Directional alignment
        · Collective motion consensus
        · Density-weighted scoring"]:::agent
    end

    RE["🛡 Reliability Estimator
    Dynamic per-frame
    weight scaling"]:::reliability

    FE(["Fusion Engine
    Weighted Integrity Score"]):::output

    IN --> A1 & A2 & A3
    A1 & A2 & A3 --> RE
    RE --> FE

    classDef input fill:#1a1a2e,stroke:#4fc3f7,color:#4fc3f7,font-weight:bold
    classDef agent fill:#161b22,stroke:#6e40c9,color:#c9d1d9,stroke-width:1.5px
    classDef reliability fill:#161b22,stroke:#f9a825,color:#f9a825,stroke-width:1.5px
    classDef output fill:#1a1a2e,stroke:#00e676,color:#00e676,font-weight:bold
```

---

### ② Motion Coherence Value (MCV)

The MCV is the **core physical signal** — the rolling z-score of kinematic features constrained by a dynamic noise floor.

```mermaid
flowchart LR
    subgraph KF["  Kinematic Features (per Track)  "]
        direction TB
        f1["Velocity magnitude"]
        f2["Acceleration magnitude"]
        f3["Angular velocity (heading Δ)"]
    end

    subgraph RW["  Rolling Window — W frames  "]
        direction TB
        mu["μ = rolling mean"]
        sigma["σ = rolling std dev"]
    end

    subgraph MCV["  MCV Formula  "]
        formula["z = (x − μ) / max(σ, ε)
        ── ε = dynamic noise floor ──
        prevents score explosion
        from micro-jitter"]:::formula
    end

    out(["MCV z-score
    High → Tamper Signal
    Low → Coherent Motion"]):::verdict

    KF --> RW --> MCV --> out

    classDef formula fill:#161b22,stroke:#f9a825,color:#f9a825,stroke-width:1.5px,font-style:italic
    classDef verdict fill:#1a1a2e,stroke:#00e676,color:#00e676,font-weight:bold
```

**The complete MCV formula:**

$$z_t = \frac{x_t - \mu_W}{\max\!\left(\sigma_W,\; \epsilon\right)}$$

| Symbol | Meaning |
|:---:|:---|
| $x_t$ | Current feature value at frame $t$ |
| $\mu_W$ | Rolling mean over window $W$ |
| $\sigma_W$ | Rolling standard deviation over window $W$ |
| $\epsilon$ | Dynamic noise floor — prevents false alarms from micro-jitter |

> [!NOTE]
> A **high** MCV z-score signals a kinematic discontinuity — a physical impossibility that strongly correlates with temporal tampering.

---

### ③ Dynamic Reliability-Aware Fusion

Agent weights are **dynamically rescaled per frame** — unreliable agents are automatically attenuated before the fusion stage.

$$\text{Effective Weight}_i = \text{Base Weight}_i \times \text{Reliability}_i$$

```mermaid
flowchart TD
    subgraph Conditions["  Runtime Conditions → Reliability Impact  "]
        direction LR
        r1["🔴 Freshly initialized track → ↓ Motion Agent weight"]
        r2["🔴 Low-confidence detection IoU < τ → ↓ Continuity Agent weight"]
        r3["🔴 Sparse crowd, N objects < threshold → ↓ Crowd Agent weight"]
        r4["🟢 Long-running stable track → ↑ All agent weights"]
    end

    FE["Fusion Engine
    Applies rescaled weights → Σ Integrity Score"]:::fusion

    Conditions --> FE

    classDef fusion fill:#161b22,stroke:#4fc3f7,color:#4fc3f7,stroke-width:1.5px,font-weight:bold
```

> [!WARNING]
> Without this mechanism, a low-confidence agent on a sparse frame would **corrupt the verdict** — the reliability estimator prevents this entirely.

---

### ④ Random Forest Meta-Classifier

Per-frame integrity timelines are distilled into a **compact global feature vector**, then classified by a trained Random Forest.

$$\mathbf{F} = \bigl[\min(\text{Integrity}),\;\mu(\text{Integrity}),\;\sigma(\text{Integrity}),\;\text{Anomaly Density},\;\log(\max(\text{MCV})),\;\text{Segment Count}\bigr]$$

```mermaid
flowchart LR
    subgraph FE["  Feature Extraction  "]
        direction TB
        f1["min_integrity → worst single frame"]
        f2["mean_integrity → overall video health"]
        f3["score_volatility → temporal instability"]
        f4["anomaly_density → fraction of flagged segments"]
        f5["log(max_MCV) → peak kinematic deviation"]
        f6["segment_count → structural discontinuities"]
    end

    RF["Random Forest Classifier
    5-Fold Stratified Cross-Validation
    Macro F1 · ROC-AUC · Per-Fold Accuracy"]:::rf

    verdict(["TAMPER VERDICT
    + Confidence Probability"]):::verdict

    FE --> RF --> verdict

    classDef rf fill:#161b22,stroke:#f9a825,color:#f9a825,stroke-width:1.5px
    classDef verdict fill:#1a1a2e,stroke:#00e676,color:#00e676,font-weight:bold
```

---

## 🛠️ Tech Stack & Requirements

| Layer | Technology | Role |
|:---:|:---|:---|
| **Core Engine** | Python 3.8+, OpenCV, NumPy, SciPy | Frame decoding, matrix math, kinematic computation |
| **State Estimation** | FilterPy (Kalman Filters) | Multi-object trajectory propagation across occlusions |
| **Vision Models** | Ultralytics YOLOv8n, YOLOv8n-pose | Real-time object detection + human pose keypoint extraction |
| **Classification** | scikit-learn Random Forest | 5-fold stratified CV meta-classification of temporal features |
| **Serialization** | Joblib, Pandas | Model persistence and evaluation dataset management |
| **Dashboard** | Streamlit, Matplotlib | Interactive forensic portal and integrity chart rendering |
| **Logging & Progress** | Loguru, tqdm | Structured terminal logging and pipeline progress visualization |
| **Video Player** | HTML5 / Vanilla JS | Dual-sync player with hover pan/zoom mirroring |

---

## 📂 Codebase Structure

```
coherex-integrity/
│
├── coherex/                         ← Core architectural package
│   ├── config.py                    ← Single source of truth — thresholds & fusion params
│   │
│   ├── detection/
│   │   └── yolo_detector.py         ← YOLOv8 object detection wrapper
│   │
│   ├── tracking/
│   │   ├── manager.py               ← Multi-object track lifecycle & ID management
│   │   ├── kalman.py                ← Per-track state estimation via Kalman filtering
│   │   └── pose.py                  ← Pose association tracking routines
│   │
│   ├── trajectory/
│   │   └── store.py                 ← Persistent kinematic history per track ID
│   │
│   ├── coherence/
│   │   └── mcv.py                   ← MCV z-score computation + dynamic noise floor
│   │
│   ├── integrity/
│   │   ├── motion_agent.py          ← Agent I  : velocity / acceleration / angular analysis
│   │   ├── continuity_agent.py      ← Agent II : track lifecycle stability scoring
│   │   ├── crowd_agent.py           ← Agent III: collective directional alignment
│   │   ├── reliability.py           ← Dynamic per-frame agent weight adjustment
│   │   └── fusion_engine.py         ← Weighted multi-agent score fusion
│   │
│   └── meta/
│       ├── feature_extractor.py     ← Global temporal feature vector generation
│       └── classifier.py            ← Random Forest training, evaluation, inference
│
├── frontend/
│   └── app.py                       ← Streamlit dashboard — main entry point
│
├── scripts/
│   ├── create_tampered_videos.py    ← Synthetic tamper dataset generation
│   ├── evaluate_dataset.py          ← Batch feature extraction across dataset
│   ├── train_meta_classifier.py     ← RF training with stratified cross-validation
│   └── run_pipeline.py              ← Isolated single-file forensic analysis
│
├── data/
│   ├── raw_videos/                  ← Baseline authentic video inputs
│   ├── models/                      ← Persisted classifier and scaler (joblib)
│   └── evaluation/                  ← results_test.csv — classifier training data
│
├── docs/images/                     ← Portal screenshots and documentation assets
├── requirements.txt
├── setup.py
└── README.md
```

---

## 🖥️ Dashboard Portal

<div align="center">
<img src="docs/images/portal_upload.png" alt="Upload Interface" width="80%" style="border-radius:10px; box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.4);"/>

*The upload and configuration panel of the CoheRex forensic dashboard — `http://localhost:8501`*
</div>

<br/>

The dashboard provides **five integrated analysis panels**:

| Panel | Functionality |
|:---:|:---|
| **① Upload Interface** | Video upload + temporal window parameters + agent weight overrides |
| **② Dual Sync Player** | Frame-accurate original vs. annotated playback with hover pan/zoom mirroring |
| **③ Integrity Timeline** | Continuous score plot with flagged segment shading and hover-to-seek |
| **④ Score Histogram** | Frame-level distribution exposing bimodal splits characteristic of splice boundaries |
| **⑤ Verdict Panel** | AI classifier output with confidence probability, statistics, and MCV violation log |

---

## 🚀 Setup & Execution

> [!TIP]
> Use a pristine virtual environment to prevent dependency conflicts with other ML frameworks.

### Step I — Environment Setup

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate       # Linux / macOS
venv\Scripts\activate          # Windows

# Install all dependencies and link the local package
pip install -r requirements.txt
pip install -e .
```

### Step II — Dataset Preparation

```bash
# Generate tampered variants from authentic baseline videos
python scripts/create_tampered_videos.py
```

Synthesizes frame deletions, duplications, speed changes, splices, and reversal clips from baseline inputs, producing a fully labeled evaluation dataset.

### Step III — Batch Evaluation

```bash
# Extract temporal feature vectors across the full dataset
python scripts/evaluate_dataset.py
```

Outputs `data/evaluation/results_test.csv` — per-video feature vectors with ground-truth labels.

### Step IV — Train the Meta-Classifier

```bash
# Train the Random Forest using 5-fold stratified cross-validation
python scripts/train_meta_classifier.py --csv data/evaluation/results_test.csv
```

Saves `data/models/meta_classifier.pkl`. Reports per-fold accuracy, macro F1, and ROC-AUC.

### Step V — Launch the Dashboard

```bash
streamlit run frontend/app.py
```

Navigate to **`http://localhost:8501`** to begin forensic investigations.

---

## 🧠 Design Philosophy

> [!IMPORTANT]
> **CoheRex is built on a single core thesis: tampering breaks physics before it breaks pixels.**

```mermaid
flowchart LR
    subgraph Pixel["  Traditional Forensics  "]
        direction TB
        p1["Pixel-level artifact analysis"]
        p2["Defeated by: re-encoding,\ncolor grading, compression"]
        p1 --> p2
    end

    subgraph Trajectory["  CoheRex Approach  "]
        direction TB
        t1["Trajectory-domain physics analysis"]
        t2["Resilient to: encoding changes,\ncamera noise, adversarial compression"]
        t1 --> t2
    end

    Pixel -- "❌ Fragile" --> X(["Tamper Detection"])
    Trajectory -- "✅ Robust" --> X

    classDef default fill:#0d1117,stroke:#30363d,color:#c9d1d9
```

Compression artifacts, color grading, and resolution changes can disguise pixel-level edits — but **no post-processing pipeline can reconstruct coherent kinematic trajectories** for objects that weren't there, or smooth over the Newtonian impossibilities introduced by frame deletion.

The multi-agent architecture follows from this: different physical properties — velocity coherence, track lifecycle integrity, collective motion consensus — are maximally informative under different scene conditions. Decoupling the agents and fusing them through a reliability layer means each contributes exactly as much as it can be trusted to, **and no more**.

---

<div align="center">
<br/>
<sub>Crafted for the advancement of computational video forensics.</sub>
</div>
