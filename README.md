#  Car Price Prediction using Machine Learning

## Project Overview
This project aims to predict car selling prices using Machine Learning Regression techniques.
The model analyzes various car-related features such as present price,
manufacturing year, kilometers driven, fuel type, transmission type, selling type,
and ownership details to estimate the expected selling price of a vehicle.
This project demonstrates a complete machine learning workflow including data preprocessing, 
feature engineering, model training, evaluation, and visualization.

## Objective
The main objective of this project is to build an accurate regression model 
that can predict the selling price of a car based on historical data and different vehicle characteristics.

## Dataset Information
The dataset contains the following features:
- Car_Name
- Year
- Selling_Price (Target Variable)
- Present_Price
- Driven_kms
- Fuel_Type
- Selling_type
- Transmission
- Owner

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- VS Code

## Project Workflow

### Data Preprocessing
The dataset was loaded using Pandas and checked for missing values and inconsistencies. 
Unnecessary columns were removed and categorical variables were converted into numerical format.

### Feature Engineering
A new feature called Car_Age was created from the Year column to improve model performance.

Formula:
Car_Age = 2026 - Year

### Model Building
The dataset was divided into training and testing sets.
A Random Forest Regression model was trained to learn relationships between car features and selling price.

### Model Evaluation
The model performance was measured using:
- R² Score
- Mean Absolute Error (MAE)

### Data Visualization
Matplotlib was used to visualize Actual vs Predicted prices and understand prediction performance.

## Results
The trained model successfully predicted car prices with good accuracy.

Example Performance:
- R² Score: 0.9585315540912656
- MAE: 0.6494213114754099

## Real-World Applications
- Used Car Price Prediction
- Online Vehicle Selling Platforms
- Automobile Market Analysis
- Insurance Price Estimation
- Vehicle Valuation Systems

## Conclusion
This project helped in understanding how machine learning regression models
can solve real-world business problems through prediction.
It also provided practical experience in data preprocessing,
feature engineering, model training, evaluation, and visualization.

##built by
Sanju 
B.Tech CSE Student  
