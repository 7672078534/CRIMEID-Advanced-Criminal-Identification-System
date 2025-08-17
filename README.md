# CRIMEID-Advanced-Criminal-Identification-System
# Overview
This project analyzes the relationship between crime rates and socioeconomic factors using machine learning regression models. It aims to predict crime rates based on features like poverty rate, education level, and regional data.

# Dataset
- Training Data: crime_vs_socioeconomic_factors.csv
- Test Data: test_data.csv
- Target Variable: Crime_Rate
- Features: Region, Poverty_Rate, and other socioeconomic indicators

# Technologies Used
- Languages: Python
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, joblib
- Models:
- K-Nearest Neighbors Regressor
- Decision Tree Regressor
- Preprocessing:
- Label Encoding for categorical features
- Polynomial Feature Expansion
- MinMax Scaling

# Workflow
1. Data Preprocessing
- Checked for nulls, duplicates, and data types
- Encoded categorical variables (Region)
- Scaled target variable (Crime_Rate) using MinMaxScaler
2. Feature Engineering
- Applied PolynomialFeatures to capture non-linear relationships
3. Model Training
- Split data into training and test sets (80/20)
- Trained and saved models using joblib
- Models stored in /model/ directory
4. Evaluation Metrics
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score
5. Prediction
- Loaded test data
- Transformed using same polynomial features
- Predicted crime rates using trained models

# Visualization
- Scatter plots for feature relationships
- Heatmap for correlation analysis
- Prediction vs Actual plots for model performance

# File Structure
├── crime_vs_socioeconomic_factors.csv
├── test_data.csv
├── model/
│   ├── KNNRegressor.pkl
│   └── DecisionTreeRegressor.pkl
├── main.py
└── README.md



# How to Run
- Clone the repository or download the files.
- Ensure required libraries are installed:
pip install pandas numpy matplotlib seaborn scikit-learn joblib
- Run the script:
python main.py
- Check predictions in the test DataFrame.

# Future Enhancements
- Integrate more models like SVR, Random Forest, or XGBoost
- Add hyperparameter tuning
- Deploy as a Flask web app for real-time predictions


