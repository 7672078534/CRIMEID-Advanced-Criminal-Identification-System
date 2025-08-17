# Step 1: Importing the Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder
# Ignore all warnings
warnings.filterwarnings('ignore')
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC,SVR
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split
# Step 2: Reading the Dataset
data = pd.read_csv("C:\\Users\\venka\\OneDrive\\Desktop\\Projects\\3-2\\3 CrimeID\\crime_vs_socioeconomic_factors.csv")
data
# Step 3: Data Preprocessing
data.shape
data.size
data.info()
data.describe()
data.columns
data.isnull().sum()
data.dtypes
data.duplicated().sum()
data.corr()
Step 4: Exploratory Data Analysis(EDA)
'''plt.figure(figsize=(12, 6))
sns.barplot(x='Region', y='Crime_Rate', data=data)
plt.xticks(rotation=45)
plt.title("Crime Rate by Region")
plt.tight_layout()
plt.show()'''
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Poverty_Rate', y='Crime_Rate', data=data)
plt.title("Crime Rate vs. Poverty Rate")
plt.xlabel("Poverty Rate (%)")
plt.ylabel("Crime Rate")
plt.show()
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data['Region'] = le.fit_transform(data['Region'])
data['Region'] = LabelEncoder().fit_transform(data['Region'])
data.head()
Step 5: X,y Separation
X = data.drop('Crime_Rate', axis=1)
X
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
y = data['Crime_Rate']
from sklearn.preprocessing import MinMaxScaler

y = data['Crime_Rate'].values.reshape(-1, 1)  # Reshape for sklearn
scaler_y = MinMaxScaler(feature_range=(0, 1))  # Set range to [0, 1]
y = scaler_y.fit_transform(y)
y #After Scaling
# Splitting the Dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_poly,y,test_size = 0.2,random_state = 42)
print('X shape is',X.shape)
print('y shape is',y.shape)
print('X_train shape is',X_train.shape)
print('X_test shape is',X_test.shape)
print('y_train shape is',y_train.shape)
print('y_test shape is',y_test.shape)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae_list, mse_list, rmse_list, r2_list = [], [], [], []

def calculateMetrics(algorithm, predict, testY):
    # Flatten arrays to ensure compatibility
    testY = testY.ravel()
    predict = predict.ravel()

    mae = mean_absolute_error(testY, predict)
    mse = mean_squared_error(testY, predict)
    rmse = np.sqrt(mse)
    r2 = r2_score(testY, predict)

    mae_list.append(mae)
    mse_list.append(mse)
    rmse_list.append(rmse)
    r2_list.append(r2)

    print(f"{algorithm} MAE: {mae:.2f}")
    print(f"{algorithm} MSE: {mse:.2f}")
    print(f"{algorithm} RMSE: {rmse:.2f}")
    print(f"{algorithm} R²: {abs(r2):.2f}")

    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=testY, y=predict, alpha=0.6)
    plt.plot([testY.min(), testY.max()], [testY.min(), testY.max()], 'r--')
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title(f"{algorithm} Prediction vs Actual")
    plt.grid(True)
    plt.show()
from sklearn.neighbors import KNeighborsRegressor
import joblib
import os

if os.path.exists('model/KNNRegressor.pkl'):
    KNN = joblib.load('model/KNNRegressor.pkl')
    print("Model Loaded Successfully.")
    predict = KNN.predict(X_test)
    calculateMetrics("KNNRegressor", predict, y_test)
else:
    KNN = KNeighborsRegressor(n_neighbors=5)  # You can tune n_neighbors
    KNN.fit(X_train, y_train)
    joblib.dump(KNN, 'model/KNNRegressor.pkl')
    print("Model Saved Successfully.")
    predict = KNN.predict(X_test)
    calculateMetrics("KNNRegressor", predict, y_test)
from sklearn.tree import DecisionTreeRegressor
import joblib
import os

if os.path.exists('model/DecisionTreeRegressor.pkl'):
    DTR = joblib.load('model/DecisionTreeRegressor.pkl')
    print("Model Loaded Successfully.")
    predict = DTR.predict(X_test[0:40])
    calculateMetrics("DecisionTreeRegressor", predict, y_test[0:40])
else:
    DTR = DecisionTreeRegressor(max_depth=8)
    DTR.fit(X_train, y_train.ravel())
    joblib.dump(DTR, 'model/DecisionTreeRegressor.pkl')
    print("Model Saved Successfully.")
    predict = DTR.predict(X_test)
    calculateMetrics("DecisionTreeRegressor", predict, y_test)
test = pd.read_csv("C:\\Users\\venka\\OneDrive\\Desktop\\Projects\\3-2\\3 CrimeID\\test_data.csv")
test
test_poly = poly.transform(test)
predict = DTR.predict(test_poly)
predict
test['Predicted'] = predict
test
