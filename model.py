import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder



df = pd.read_csv("student_performance_large_dataset.csv")  # Update with your actual file name
print(df.head())  # View first 5 rows 
print(df.info())  # Check for missing values & data types  
print(df.describe())  # Summary statistics

print("====================missing values==============")
print(df.isnull().sum())  # Count missing values in each column

# encode the categorical features step by step


features = [
    "Study_Hours_per_Week", 
    "Preferred_Learning_Style", 
    "Attendance_Rate (%)", 
    "Assignment_Completion_Rate (%)",
    "Self_Reported_Stress_Level", 
    "Time_Spent_on_Social_Media (hours/week)", 
    "Sleep_Hours_per_Night"
]

X = df[features]  # Independent variables

y = df["Final_Grade"]  # Dependent variable (exam performance)

