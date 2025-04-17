# Predicting Housing Prices Using Supervised Machine Learning with Python

In this regression project, I built a supervised machine learning model to predict housing prices based on a variety of features.

## Situation

Similar to my other project documented in the repository `housing-value-classification-supervised-ml-python`, this project uses a dataset of historical home prices in Ames, Iowa. The goal here is to build a regression model that predicts the actual sale price of a home based on characteristics such as garage size, year built, building type, and more.

This project was part of a regression competition on Kaggle, where models were evaluated against actual housing prices using the Root Mean Squared Error (RMSE) metric. The final model I submitted achieved an RMSE of **28,454.19** and a Kaggle score of **0.13481**.

## Approach

After gaining a better understanding of the housing market and identifying the most impactful features for pricing, I explored and preprocessed the data.

To evaluate performance, I used several regression metrics:  
- Mean Absolute Error (MAE)  
- Root Mean Squared Error (RMSE)  
- Mean Absolute Percentage Error (MAPE)  
- R-squared (R²)

I trained and evaluated the following regression models using scikit-learn:  
- `GradientBoostingRegressor`  
- `RandomForestRegressor`  
- `LinearRegression`  
- `HistGradientBoostingRegressor`  
- `Lasso`  
- `BayesianRidge`

For feature selection, I experimented with:  
- Recursive Feature Elimination with Cross-Validation (RFECV)  
- Variance Threshold  
- SelectKBest

The best results were achieved using the `GradientBoostingRegressor`, in combination with RFECV (using `GradientBoostingRegressor` as the estimator), Variance Threshold (set to 0), and hyperparameter tuning with `RandomizedSearchCV`.

## Files

### Data
- `train.csv`: Training data  
- `test.csv`: Test data  

### Script
- `Regression_ML_GradientBoostingRegressor.ipynb`: Notebook documenting the full process using the GradientBoostingRegressor model.

### Document
- `data_description.txt`: Contains descriptions of each feature in the housing dataset.

## Using the Files

1. Download the data files and notebook and save them to your Google Drive.  
2. Update the file paths in the notebook as needed.  
3. Run the notebook in Google Colab or Jupyter to reproduce the analysis.

## Languages and Libraries

- Python 3.10.12  
- pandas 2.2.2  
- scikit-learn 1.4.2  

## Tools

- Google Colab (or Jupyter Notebook)  
- Google Drive for storage
