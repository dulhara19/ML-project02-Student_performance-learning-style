import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv("student_performance.csv")  # Update with your actual file name
print(df.head())  # View first 5 rows 
print(df.info())  # Check for missing values & data types
print(df.describe())  # Summary statistics
