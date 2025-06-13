---

# ConfigurableMLPipeline

## 📌 Overview  
A modular, configurable, and end-to-end machine learning pipeline for **regression tasks**, designed to automate data ingestion, preprocessing, training, evaluation, and hyperparameter tuning.   
> ⚠️ Flask integration is manual — user must manually select a model and update `app.py`.

---

## 🎯 Key Features
- Configurable via `config.json`
- Modular folder and file structure
- Supports 9 regression models
- Cross-validation and hyperparameter tuning
- Model performance logging and saving
- Manual Flask app for serving predictions
- Notebooks for EDA and experimentation

---

## 🛠️ Tech Stack
- Python 3.9  
- Libraries: `scikit-learn`, `xgboost`, `pandas`, `numpy`, `joblib`, `Flask`, `matplotlib`, `seaborn`, `logging`

---

## 📂 Project Structure
```

ML\_project/
├── app.py                     # Flask app (manual integration)
├── main.py                    # Entry point for pipeline
├── config.json                # Configuration file
├── requirements.txt           # Dependencies
│
├── data/
│   └── coffee\_shop\_revenue.csv
│
├── logs/
│   └── pipeline\_log.logs
│
├── outputs/
│   ├── tuned models/
│   └── metrics.json
│
├── notebooks/
│   ├── eda.ipynb
│   └── experiments.ipynb
│
├── templates/
│   ├── form.html
│   └── result.html
│
├── src/
│   ├── config/
│   │   └── config\_loader.py
│   ├── data/
│   │   └── load\_data.py
│   ├── models/
│   │   ├── hyperparameter\_tuning.py
│   │   ├── model\_configurater.py
│   │   └── model\_training.py
│   ├── pipelines/
│   │   └── run\_pipeline.py
│   ├── preprocessing/
│   │   ├── cross\_validation.py
│   │   ├── feature\_selection.py
│   │   ├── feature\_scaling.py
│   │   └── splitting\_data.py
│   └── utils/
│       └── logger\_code.py

````

---

## ⚙️ How It Works

1. Load dataset from path specified in `config.json`
2. Preprocess data:
   - Split into train/test sets
   - Cross-validation score to get baseline metrics
   - Feature selection
   - Cross-validation with selected features
   - Feature scaling
   - Train models and evaluate them on test data
   - Select N number of models to hyper-tune
   - Hyper-parameter tune and log and save hyper-tuned models with their metrics and best parameters

3. Train 9 models using cross-validation
4. Select top N models for hyperparameter tuning
5. Save best model(s) and evaluation metrics
6. Use Jupyter notebooks for EDA and further analysis

---

## 🧪 Supported Models
- Linear Regression  
- Ridge Regression  
- Lasso Regression  
- ElasticNet  
- Decision Tree  
- Random Forest  
- XGBoost  
- K-Nearest Neighbors  
- Support Vector Regressor

---

## 📊 Evaluation Metrics
- RMSE  
- MAE  
- R²  
- MSE

Example (Random Forest):
- `RMSE = 227.21`  
- `R² = 0.9475`  
- `MAE = 182.64`  
- `MSE = 51623.24`

---

## ⚙️ Configuration File (`config.json`) example.
```json
{
  "splitting": {
    "test_size": 0.2,
    "random_state": 3
  },
  "paths": {
    "outputs_directory": "outputs/",
    "dataset_path": "data/coffee_shop_revenue.csv"
  },
  "K_fold": {
    "n_splits": 5,
    "random_state_for_cv": 3
  },
  "feature_selection": {
    "selected_feature_selection_method": "feature_importance",
    "corr_threshold": 0.20,
    "no_of_features_to_select_using_feature_imp": 3,
    "features_to_select_k_using_mutual_info": 3
  },
  "scaling_method": {
    "scaler": "standard"
  },
  "no_of_models_to_select_for_hyperparameter_tuning": {
    "no_of_models": 2
  },
  "gridsearch_params": {
    "optimize_for": "neg_root_mean_squared_error",
    "n_jobs": -1,
    "cv": 5
  }
}
````

---

## 🚀 Getting Started

### 1. Clone the Repo

```bash
git clone the repo
cd ConfigurableMLPipeline
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```
### 3. Update the paths in the config.json file and then update the path in the config_loader.py component to direct towards the config.json file

### 4. Run the Pipeline

```bash
python main.py
```

### 4. View Outputs

* Check `outputs/` for saved models and metrics
* Check `logs/pipeline_log.logs` for step-by-step logs

---

## 🌐 Flask App (Manual Integration)

1. Choose a model and scaler from `outputs/`
2. Update file paths manually in `app.py`
3. Run:

```bash
python app.py
```

4. Open browser at localhost

---

## 📌 Notes

* Built to showcase modular pipeline design.
* Flask integration is **not** automated — this separation was intentional for flexibility and clarity.

---

## ✅ What I Learned

* End-to-end ML pipeline design
* Clean code structure and modularization
* Configuration-based architecture
* Manual vs. automated deployment trade-offs
* Logging, and output traceability

---
