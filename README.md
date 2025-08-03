# 🏠 Ames Housing Price Prediction

This project builds a robust regression model to predict housing prices in Ames, Iowa using the well-known [Kaggle dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques). The notebook features exploratory data analysis, feature engineering, regularization, boosting (LightGBM), and interpretability using SHAP.

---

## 🚀 Project Highlights

- **Goal**: Predict `SalePrice` using 80+ features (categorical & numeric)
- **Modeling**: LightGBM with log-transformed targets, optimized using Optuna
- **Explainability**: SHAP plots to identify top contributing features
- **Visual Storytelling**: Exported `.png` plots ready for portfolio use (Cursor + Tailwind UI)

---

## 📊 Key Visuals

| Visual | Description |
|--------|-------------|
| `correlation_heatmap.png` | Top correlated numeric features |
| `price_distribution_before_after_log.png` | SalePrice distribution (log transformation) |
| `lightgbm_feature_importance.png` | Top features ranked by model importance |
| `shap_summary_plot.png` | SHAP global feature contributions |
| `shap_top_features.png` | Local SHAP breakdown for selected homes |
| `residuals_vs_predicted.png` | Residual analysis to identify bias |
| `actual_vs_predicted.png` | Model performance on test predictions |

> 🎯 **Model Performance**  
> Final model achieves RMSE of **0.121** on log-transformed prices and explains over **91% of variance** (R²) on validation set.

---

## ⚙️ Technical Stack

- **Notebook**: `Ames_House_Clean_Final.ipynb`
- **Core Libraries**: pandas, seaborn, matplotlib, sklearn, LightGBM, SHAP, Optuna
- **Pipeline Flow**:
  1. Data cleaning & missing value imputation
  2. Feature encoding (ordinal, one-hot, domain-specific logic)
  3. Log-transform target + skewed numerical features
  4. Train/test split with cross-validation
  5. Optuna-based hyperparameter tuning
  6. Residuals & SHAP interpretability

---

## 🧠 Interpretation with SHAP

SHAP values show that features like:
- `OverallQual`, `GrLivArea`, `GarageCars`, and `TotalBsmtSF`
- Have the strongest influence on predictions

<insert SHAP summary image here>

Use cases include:
- Buyer/seller price justifications
- Explainable ML for lending or property valuation

---

## 🔁 Reproducibility

To run this project:

```bash
git clone https://github.com/yourname/ames_house_prices.git
cd ames_house_prices
pip install -r requirements.txt
jupyter notebook Ames_House_Clean_Final.ipynb
# Kaggle-Data-Challenge---Ames_House_Prices
