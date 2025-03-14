# 🎓 Student Performance Prediction using Linear Regression

## Introduction
This project predicts student performance based on various academic, lifestyle, and psychological factors using **Linear Regression**. The model is trained on student data and predicts final grades based on multiple independent variables.

## Dataset Overview
The dataset includes:
- **Demographic Info**: Age, Gender  
- **Study Habits**: Study Hours per Week, Online Courses Completed  
- **Engagement**: Participation in Discussions, Use of Educational Technology  
- **Performance Metrics**: Assignment Completion Rate, Exam Score, Attendance Rate  
- **Lifestyle**: Social Media Usage, Sleep Hours  
- **Psychological Factors**: Preferred Learning Style, Self-Reported Stress Level  
- **Target Variable**: `Final_Grade` (Numerical)  

## Data Preprocessing
- **Handling Missing Values**: Checked and confirmed no missing values.  
- **Encoding Categorical Features**: Used binary and one-hot encoding for categorical variables.  
- **Feature Selection**: Selected key features affecting student grades.  

## Model Training
- **Algorithm**: Linear Regression  
- **Train-Test Split**: 80% Training, 20% Testing  
- **Implementation**:
  ```python
  from sklearn.linear_model import LinearRegression

  model = LinearRegression()
  model.fit(X_train, y_train)

 
## Feature Importance

Regression coefficients analyzed to understand feature impact.
Visualization:
```python
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Coefficient", y="Feature", data=coef_df, palette="coolwarm")
    plt.axvline(0, color="black", linewidth=1.2)
    plt.title("Feature Importance (Linear Regression Coefficients)")
    plt.show()
```   

## Model Evaluation

Metrics Used:
    * Mean Squared Error (MSE)
    * R² Score
```python
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.4f}")
```


## Conclusion

This project successfully applies Linear Regression to analyze student performance, showing meaningful relationships between study habits, participation, and grades. Future improvements:

    Testing alternative models (Decision Trees, Random Forest)
    Expanding feature selection with psychological & social factors
    Applying feature scaling & polynomial regression for better performance

📌 Author: Dulhara Lakshan
📌 License: MIT
📌 Contributions: Open to improvements! Feel free to fork, modify, and create pull requests.