# AI4EAC-Liquidity-Stress-Early-Warning-Challenge
## Overview
 
This repository contains my solution to the [AI4EAC Liquidity Stress Early Warning Challenge](https://zindi.africa/competitions/liquidity-stress-early-warning-challenge), hosted on Zindi and open to university students across the East African Community.
 
**Objective:** Build a machine learning model that predicts the probability that a customer will experience liquidity stress within 30 days of an observation date.
 
## Problem Statement
 
Financial institutions in East Africa face significant challenges in identifying customers at risk of liquidity stress before it occurs. This challenge tasked participants with building an early warning system using historical customer data to output predicted probabilities of liquidity stress — enabling proactive intervention rather than reactive damage control.
 
## Evaluation Metrics
 
The competition used a weighted multi-metric evaluation:
 
| Metric | Weight | Description |
|--------|--------|-------------|
| **Log Loss** | 60% | Penalises confident incorrect predictions |
| **ROC-AUC** | 40% | Measures how well the model ranks stressed vs. non-stressed customers |
 
## Approach
 
### Feature Engineering
- Cleaned and transformed raw customer financial data
- Engineered features capturing behavioural patterns, transaction history, and temporal signals
- Identified key predictors driving liquidity stress likelihood
### Modelling
- Built an ensemble model using **XGBoost** and **LightGBM**
- Applied cross-validation to prevent overfitting and ensure generalisability
- Tuned hyperparameters to optimise the weighted Log Loss / ROC-AUC score
### Output
- Model outputs predicted **probabilities** (not binary labels) for each customer
- Probabilities indicate the likelihood of experiencing liquidity stress within 30 days
## Repository Structure
 
```
AI4EAC-Liquidity-Stress-Early-Warning-Challenge/
│
├── README.md
├── solution.py          # Main modelling script
└── requirements.txt     # Dependencies
```
 
## Requirements
 
```bash
pip install -r requirements.txt
```
 
Key libraries:
- `pandas`, `numpy` — data manipulation
- `scikit-learn` — preprocessing and evaluation
- `xgboost`, `lightgbm` — gradient boosting models
## How to Run
 
```bash
python solution.py
```
 
The script will load the data, run the full pipeline (preprocessing → feature engineering → model training → prediction), and output a submission-ready CSV file.
 
