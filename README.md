# Used Car Price Predictor 🚗

A machine learning web app that estimates the fair market price of a used car based on its specifications — built with Random Forest regression.

**🔗 Live App:** [[Add your Streamlit Cloud link here]](https://usedcarpricepredictor-2026.streamlit.app/)

---

## 📌 Overview

Pricing a used car is hard — sellers often overprice out of attachment, and buyers rarely know what's "fair." This project uses **machine learning** to remove the guesswork, estimating a car's market value from real listing data based on its year, mileage, engine specs, and condition.

Given a car's details, the app predicts an estimated price — along with a realistic price range, not a single misleading number.

## 🧠 How It Works

This project uses **Random Forest Regression**, trained and evaluated using a full scikit-learn preprocessing pipeline.

1. **Data:** 2,095 real used car listings, with features like brand, year, kilometers driven, fuel type, transmission, engine size, and mileage
2. **Preprocessing:** Built a `ColumnTransformer` pipeline — numeric features scaled with `StandardScaler`, categorical features encoded with `OneHotEncoder`
3. **Model comparison:** Evaluated Random Forest, Gradient Boosting, XGBoost, and Ridge Regression — Random Forest and Gradient Boosting performed best (R² ≈ 0.90)
4. **Deployment:** Wrapped the trained pipeline in an interactive Streamlit app, with prices converted to PKR

## 📊 Model Performance

| Metric | Value |
|---|---|
| R² Score | 0.90 |
| MAE | ~Rs. 135,000 |

The top predictors of price were **engine power**, **manufacturing year**, and **engine displacement (CC)** — consistent with real-world car valuation logic.

## 🛠️ Tech Stack

- **Python** — core language
- **scikit-learn** — Random Forest, preprocessing pipeline
- **Pandas** — data handling
- **Streamlit** — web app framework
- **joblib** — model serialization

## ⚠️ Known Limitations

- Prices were converted from INR to PKR using an approximate exchange rate (1 INR ≈ 2.88 PKR, as of July 2026). Since exchange rates fluctuate, this is an estimate rather than a live conversion.
- The model is trained on a specific regional dataset; pricing patterns may not generalize perfectly to all local markets.
- Predictions for cars with highly unusual specifications (e.g. extremely high mileage or very old manufacturing years) may be less reliable, since the model has limited training examples in those ranges.

## 🚀 Running Locally

```bash
git clone https://github.com/mshaheer7u-cloud/UsedCarPricePredictor.git
cd UsedCarPricePredictor
pip install -r requirements.txt
streamlit run app.py
```

## 📂 Project Structure

```
UsedCarPricePredictor/
├── app.py                  # Streamlit application
├── requirements.txt        # Python dependencies
├── runtime.txt              # Python version
└── car_price_model.pkl      # Trained Random Forest pipeline
```

---

*Part of a broader machine learning portfolio exploring regression, classification, imbalanced classification, and clustering across different real-world domains.*
