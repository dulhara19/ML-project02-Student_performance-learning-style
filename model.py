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

# print("====================missing values==============")
# print(df.isnull().sum())  # Count missing values in each column

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
# print(df["Final_Grade"])

# again checking encoded data
# print(df.head())

# print all the cols in updated ds
print(df.columns)  # This will show the actual column names after encoding



features = [
    "Age",
    "Gender",
    "Study_Hours_per_Week",
    "Online_Courses_Completed",
    "Participation_in_Discussions",
    "Assignment_Completion_Rate (%)",
    "Exam_Score (%)",
    "Attendance_Rate (%)",
    "Use_of_Educational_Tech",
    "Time_Spent_on_Social_Media (hours/week)",
    "Sleep_Hours_per_Night",

    # One-hot encoded learning style columns
    "Preferred_Learning_Style_Kinesthetic",
    "Preferred_Learning_Style_Reading/Writing",
    "Preferred_Learning_Style_Visual",

    # One-hot encoded stress level columns
    "Self_Reported_Stress_Level_Low",
    "Self_Reported_Stress_Level_Medium"
]


X = df[features]  # Independent variables

y = df["Final_Grade"]  # Dependent variable (exam performance)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Print model coefficients
print(f"Intercept (c): {model.intercept_}")
print(f"Coefficients (m): {model.coef_}")

# Creating a DataFrame for visualization
coef_df = pd.DataFrame({"Feature": features, "Coefficient": model.coef_})

# Sorting by absolute value of coefficient (important features on top)
coef_df = coef_df.reindex(coef_df["Coefficient"].abs().sort_values(ascending=False).index)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x="Coefficient", y="Feature", data=coef_df, palette="coolwarm")
plt.axvline(0, color="black", linewidth=1.2)  # Vertical line at 0 for reference
plt.title("Feature Importance (Linear Regression Coefficients)", fontsize=14)
plt.xlabel("Coefficient Value", fontsize=12)
plt.ylabel("Features", fontsize=12)
plt.show()

# Loop through each feature and plot Final_Grade vs Feature
for feature in features:
    plt.figure(figsize=(6, 4))  # Set figure size for each plot
    
    sns.scatterplot(x=df[feature], y=df["Final_Grade"], alpha=0.5, color="blue")
    
    plt.xlabel(feature)
    plt.ylabel("Final_Grade")
    plt.title(f"Final Grade vs {feature}")
    
    plt.show()  # Show one plot at a time