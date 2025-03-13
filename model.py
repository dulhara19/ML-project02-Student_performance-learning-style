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
# i have already updated the csv file as df 

# Creating LabelEncoder object
label_encoder = LabelEncoder()

# Encode binary categorical variables
df["Gender"] = label_encoder.fit_transform(df["Gender"])  # Female → 0, Male → 1
df["Participation_in_Discussions"] = label_encoder.fit_transform(df["Participation_in_Discussions"])  # No → 0, Yes → 1++++++++++
df["Use_of_Educational_Tech"] = label_encoder.fit_transform(df["Use_of_Educational_Tech"])  # No → 0, Yes → 1

# One-hot encoding for multi-class categorical variables
df = pd.get_dummies(df, columns=["Preferred_Learning_Style", "Self_Reported_Stress_Level"], drop_first=True)

#Encoding the Target Variable (Final_Grade)
df["Final_Grade"] = label_encoder.fit_transform(df["Final_Grade"])

# Show the 'Final_Grade' column
print(df["Final_Grade"])

# again checking encoded data
print(df.head())

# print all the cols in updated ds
print(df.columns)


features = [
    "Age",
    "Gender",
    "Study_Hours_per_Week",
    "Preferred_Learning_Style",
    "Online_Courses_Completed",
    "Participation_in_Discussions",
    "Assignment_Completion_Rate (%)",
    "Exam_Score (%)",
    "Attendance_Rate (%)",
    "Use_of_Educational_Tech",
    "Self_Reported_Stress_Level",
    "Time_Spent_on_Social_Media (hours/week)",
    "Sleep_Hours_per_Night"
]

X = df[features]  # Independent variables

y = df["Final_Grade"]  # Dependent variable (exam performance)

