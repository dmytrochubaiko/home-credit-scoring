# 🏦 Home Credit Default Risk — End-to-End Scoring Service

An end-to-end Machine Learning project designed to assess borrower creditworthiness and predict the probability of loan default. This repository covers the complete ML lifecycle: from exploratory data analysis (EDA), feature engineering, and hyperparameter tuning to deploying a stacked ensemble model as a containerized FastAPI microservice.

---

## 📌 Key Features

- **Ensemble Architecture (Out-of-Fold Stacking):** Combines cross-validated Gradient Boosting models (**LightGBM** and **XGBoost**) with a Meta-Model blender to maximize predictive performance (ROC-AUC).
- **Business Risk Grading:** Automatically translates raw default probabilities into actionable credit risk tiers (`A_Low_Risk`, `B_Moderate_Risk`, `C_High_Risk`, `D_Default_Likely`).
- **Modular Preprocessing Pipeline:** Custom feature engineering and missing value imputation designed to prevent data leakage during inference.
- **FastAPI Microservice:** High-performance REST API supporting both **single-client real-time scoring** and **batch processing**, validated via Pydantic schemas.
- **Docker Containerization:** Fully isolated environment with all system dependencies (`libgomp1`) and pre-trained model weights ready for instant deployment.

---

## 🏗 Project Structure

```text
home-credit-scoring/
│
├── Models/                                  # Serialized trained models (.joblib)
│   ├── lgbm_folds.joblib                    # Cross-validated LightGBM models
│   ├── xgb_folds.joblib                     # Cross-validated XGBoost models
│   └── meta_model.joblib                    # Stacking meta-estimator
│
├── Notebooks/                               # Research & experimentation pipeline
│   ├── 01_data_preroration.ipynb            # EDA, feature engineering & aggregation
│   ├── 02_preparation_tuning.ipynb       # Optuna/GridSearch model optimization
│   └── 03_oof_blend_submission.ipynb        # Out-of-Fold training & stacking ensemble
│
├── Production/                              # Production inference & API layer
│   ├── app.py                               # FastAPI application & endpoints
│   ├── pipeline.py                          # Feature engineering & preprocessing logic
│   └── predict.py                           # CreditScoringService inference engine
│
├── Dockerfile                               # Container build instructions
├── requirements.txt                         # Production Python dependencies
└── README.md                                # Project documentation
```

---

## 🚀 Quick Start

### Option 1: Run with Docker (Recommended)

Make sure you have [Docker](https://www.docker.com/) installed and running on your machine.

```bash
# 1. Build the Docker image
docker build -t credit-scoring-api:v1 .

# 2. Run the container in detached mode
docker run -d -p 8000:8000 --name credit_scoring credit-scoring-api:v1
```

Once the container is running:
- **API Base URL:** `http://localhost:8000`
- **Interactive Swagger UI Documentation:** `http://localhost:8000/docs`

---

### Option 2: Local Setup (Without Docker)

```bash
# 1. Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the FastAPI server
uvicorn Production.app:app --host 0.0.0.0 --port 8000 --reload
```

---

## 📡 API Usage Examples

### 1. Health Check
**`GET /`**
```json
{
  "message": "Credit Scoring API is running! Use /predict endpoint for risk assessment."
}
```

### 2. Single Client Prediction
**`POST /predict`**

**Request Payload:**
```json
{
  "data": {
    "SK_ID_CURR": 100002,
    "AMT_INCOME_TOTAL": 202500.0,
    "AMT_CREDIT": 406597.5,
    "AMT_ANNUITY": 24700.5,
    "DAYS_BIRTH": -9461,
    "DAYS_EMPLOYED": -637
  }
}
```

**Response:**
```json
{
  "status": "success",
  "prediction": {
    "SK_ID_CURR": 100002,
    "probability": 0.284,
    "risk_grade": "C_High_Risk"
  }
}
```

### 3. Batch Prediction (Multiple Clients)
**`POST /predict`**

**Request Payload:**
```json
{
  "data": [
    {
      "SK_ID_CURR": 100002,
      "AMT_INCOME_TOTAL": 202500.0,
      "AMT_CREDIT": 406597.5
    },
    {
      "SK_ID_CURR": 100003,
      "AMT_INCOME_TOTAL": 270000.0,
      "AMT_CREDIT": 1293502.5
    }
  ]
}
```

**Response:**
```json
{
  "status": "success",
  "prediction": [
    {
      "SK_ID_CURR": 100002,
      "probability": 0.284,
      "risk_grade": "C_High_Risk"
    },
    {
      "SK_ID_CURR": 100003,
      "probability": 0.032,
      "risk_grade": "A_Low_Risk"
    }
  ]
}
```

---

## 🛠 Tech Stack

- **Machine Learning:** LightGBM, XGBoost, Scikit-Learn, Pandas, NumPy, Joblib
- **API & Validation:** FastAPI, Uvicorn, Pydantic
- **Deployment:** Docker (`python:3.10-slim`)
