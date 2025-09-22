# Workout Analysis

##  Project Overview  
This project explores how **machine learning can model fitness performance** using real-world workout logs (my own).  
It is divided into two main phases:  

1. **Calories Burned Prediction (Cardio Data)**  
2. **Strength Performance Prediction (Resistance Training Data)**  

A third phase (**Daily Total Training Load Prediction**) was explored but showed the challenges of combining fundamentally different modalities (strength vs cardio).  

The goal was not just predictive accuracy, but **extracting insights** about what factors drive calorie burn and strength performance.  

---

## Data Sources  
- **Cardio Dataset (`df_cardio`)**  
  - Features: `date, exercise, duration, distance, calories, month`  
  - 244 entries (64% missing calories → imputed)  

- **Strength Dataset (`df_strength`)**  
  - Features: `date, body_part, exercise, set_no, weight_kg, reps, volume_kg`  
  - ~5.7k entries  

---

## Phase 1 – Calories Burned Prediction  

### Problem Statement  
Can we predict calories burned in a cardio session using workout metrics (duration, distance, pace, exercise type, seasonality)?  

### Methods  
- **Data Cleaning**  
  - Removed redundant fields (`body_part`, `notes`)  
  - Normalized exercise names (e.g., `cycle` → `cycling`)  
  - Imputed missing calories using regression model  
- **Feature Engineering**  
  - `pace` = distance ÷ duration  
  - Cyclical month encoding (`sin_month`, `cos_month`)  
  - One-hot encoding for exercise type  
- **Models**  
  - Linear Regression, Random Forest, XGBoost  

### Results  
- **Best Model**: XGBoost (R² ≈ 0.41, MAE ≈ 6.6, RMSE ≈ 95)  
- **Key Insights**  
  - **Intensity (pace)** is the strongest driver of calorie burn  
  - **Duration and seasonality** also matter  
  - **Exercise type** mattered far less once pace/duration were included  

---

## Phase 2 – Strength Performance Prediction  

### Problem Statement  
Can we predict next-day strength performance (total training volume) using past training logs?  

### Methods  
- **Data Transformation**  
  - Aggregated daily total volume  
  - Engineered lag and rolling features:  
    - `strength_lag1` (yesterday’s volume)  
    - `strength_rolling7` (weekly load)  
    - `strength_rolling30` (monthly load)  
  - Target: next-day training volume  
- **Models**  
  - Linear Regression, Random Forest, XGBoost  

### Results  
- **Predictive Accuracy**: All models had poor numeric performance (R² < 0)  
- **Key Insights**  
  - **Weekly workload (7-day rolling)** is the dominant predictor of performance  
  - **Single-day load** and **long-term 30-day load** play smaller but real roles  
  - Results align with exercise science concepts (acute vs chronic workload balance)  

---

## Total Training Load Prediction (Exploratory)  

### Goal  
Combine strength and cardio into a unified workload metric and predict next-day total load.  

### Approach  
- Defined:  

  \[
  total\_load = strength\_volume + \alpha \times cardio\_calories
  \]  

- Engineered lag/rolling features as in Phase 2  
- Applied same models (LR, RF, XGBoost)  

### Outcome  
- Models failed to generalize (R² < 0)  
- Cardio contributed very little relative to strength (due to imbalance in records)  
- **Insight**: Simple weighted sums are insufficient to unify modalities — real workload modeling requires domain calibration (e.g., METs, heart rate, recovery data)  

---

## Conclusion  
- **Phase 1**: Moderate predictive accuracy, strong insight → pace & duration drive calorie burn  
- **Phase 2**: Poor predictive accuracy, but valuable insight → weekly workload consistency is the key driver of strength performance  
- **Phase 3**: Demonstrated limitations of combining strength + cardio without physiological calibration  

**Big takeaway:** Even when models fail to predict precisely, **data science can still extract actionable insights** about fitness patterns.  

---

## Tech Stack  
- **Python**: Pandas, NumPy, Matplotlib, Scikit-learn, XGBoost  
- **ML Methods**: Regression models, pipelines, feature engineering  
- **EDA & Visualization**: Area plots, rolling averages, feature importance  

---

## Future Work  
- Incorporate **body part splits** for strength data  
- Use **heart rate or wearable data** for cardio load  
- Explore **sequence models (LSTM, Temporal CNNs)** for time series prediction  
- Investigate **acute:chronic workload ratio (ACWR)** as a predictor of performance  

---
