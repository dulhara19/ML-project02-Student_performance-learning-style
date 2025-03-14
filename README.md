# ML-project02-student performance and learning style

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

 
