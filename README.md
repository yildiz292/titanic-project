# titanic-project

Titanic Survival Prediction Project

This project analyzes the famous *Titanic dataset* and builds a machine learning model to predict whether a passenger survived or not. The dataset is part of the Kaggle Titanic competition and is commonly used for data analysis and model training practice.

Project Overview

The goal of this project is to explore, clean, and analyze the Titanic dataset, then build a predictive model using machine learning.

Main Steps:
1. Data Cleaning
   - Filled missing values in the Age and Embarked columns.  
   - Removed unnecessary columns such as Cabin if too many values were missing.  

2. Feature Engineering
   - Converted categorical variables (Sex, Embarked) into numeric form using pd.get_dummies().  
   - Created new binary features (e.g., 0 for male, 1 for female).  

3. Model Training
   - Used algorithms such as *Logistic Regression* (or whichever you used) to train the model.  
   - Split the dataset into training and testing sets.  

4. Evaluation
   - Calculated model accuracy using metrics like accuracy_score.  

Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib / Seaborn
- Scikit-learn

Example Insights
- Female passengers had a higher survival rate.
- Passengers in higher classes (1st class) were more likely to survive.
- Younger passengers tended to survive more often

Author
Yıldız Dağdeviren
Management Information Systems Student  
Passionate about Data Analysis and Machine Learning  
