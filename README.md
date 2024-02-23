# Stroke Prediction Project

## Overview

This repository contains the code and documentation for a machine learning project aimed at predicting the likelihood of stroke based on various features. The dataset includes features such as age, hypertension, average glucose level, heart disease, BMI, smoking, and the target variable indicating the presence of a stroke, particularly focusing on users with heart disease.

## Table of Contents

- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data Exploration](#data-exploration)
- [Data Preprocessing](#data-preprocessing)
- [Model Building](#model-building)
- [Evaluation](#evaluation)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)

## Dataset

**The dataset used in this project contains the following features:**

- Age
- Hypertension
- Average Glucose Level
- Heart Disease
- BMI
- Smoking
- Stroke (target variable)

The focus of the prediction is on users with heart disease.

## Libraries

- `pandas`: Used for data manipulation and analysis.
- `numpy`: Used for numerical operations.
- `os`: Provides a way of using operating system-dependent functionality.
- `matplotlib.pyplot` and `seaborn`: Used for data visualization, including plotting correlation matrices.
- `sklearn.preprocessing`: Used for data preprocessing, including `FunctionTransformer`, `OneHotEncoder`, and `StandardScaler`.
- `imblearn.over_sampling`: Used for handling imbalanced datasets, specifically `RandomOverSampler`.
- `sklearn.decomposition`: Used for dimensionality reduction, specifically `PCA`.
- `sklearn.neighbors`: Implements the k-nearest neighbors algorithm (`KNeighborsClassifier`).
- `sklearn.linear_model`: Implements logistic regression (`LogisticRegression`).
- `sklearn.naive_bayes`: Implements the Gaussian Naive Bayes algorithm (`GaussianNB`).
- `sklearn.ensemble`: Implements random forest (`RandomForestClassifier`), AdaBoost (`AdaBoostClassifier`), and Gradient Boosting (`GradientBoostingClassifier`).
- `sklearn.svm`: Implements Support Vector Machines (`SVC`).
- `sklearn.tree`: Implements decision tree classifier (`DecisionTreeClassifier`).
- `tensorflow` and `tensorflow.keras`: Used for building and training neural networks.

## Machine Learning Models

- **K-Nearest Neighbors:** `KNeighborsClassifier` (named `knn_model`).
- **Logistic Regression:** `LogisticRegression`.
- **Naive Bayes:** `GaussianNB`.
- **Random Forest:** `RandomForestClassifier`.
- **Support Vector Machine:** `SVC`.
- **Decision Tree:** `DecisionTreeClassifier`.
- **Neural Network:** Implemented using `tensorflow.keras.Sequential` with `Dense` layers.

## Data Preprocessing and Pipeline

- `StandardScaler`: Used for feature scaling.
- `FunctionTransformer`: Allows the application of a custom function during the transformation.
- `OneHotEncoder`: Used for one-hot encoding categorical variables.
- `Pipeline`: Used for creating a data processing pipeline.

# Stroke Prediction Project

## Data Exploration
![image](https://github.com/priyanshu1947/Stroke_Prediction/assets/70458921/21683dd6-a97b-40d7-af29-9465ad61e3fa)

### Check for missing values

- Explore and handle missing values appropriately.
  ![image](https://github.com/priyanshu1947/Stroke_Prediction/assets/70458921/899ea0b2-10c6-4e17-bf29-0fb45c6b0819)

### Separate Categorical and Numerical Features

- Identify and categorize features as categorical or numerical.
  ![image](https://github.com/priyanshu1947/Stroke_Prediction/assets/70458921/971f1936-7ba0-4597-b0e8-77a7e5380699)


### Check for Duplicate Values

- Examine the dataset for duplicate entries.
  ![image](https://github.com/priyanshu1947/Stroke_Prediction/assets/70458921/8d51a5a5-c9b3-4d8c-a219-7575a18bc727)


### Visualize Categorical Features
![image](https://github.com/priyanshu1947/Stroke_Prediction/assets/70458921/1fe525b6-8c3d-4fb1-a04d-23d8e8536102)
![image](https://github.com/priyanshu1947/Stroke_Prediction/assets/70458921/afcf2ba0-b18e-4caf-a554-df8acbd4f425)
![image](https://github.com/priyanshu1947/Stroke_Prediction/assets/70458921/a1898166-9c7f-4cc9-b3d9-74cb8b9a120a)
![image](https://github.com/priyanshu1947/Stroke_Prediction/assets/70458921/7a0cfc2b-5f5a-4a10-822b-49428ebe89bf)



- Utilize visualizations to gain insights into categorical features.

## Handling Missing Values
![image](https://github.com/priyanshu1947/Stroke_Prediction/assets/70458921/a74e1db1-bffd-4888-980a-402d7a50a510)


- Implement appropriate strategies for handling missing values.

## Drop Irrelevant Feature
![image](https://github.com/priyanshu1947/Stroke_Prediction/assets/70458921/b3be1614-bdfc-4f05-be71-5a80b1b50755)


- Drop irrelevant features, such as 'id'.

## Convert Categorical Features into Numerical
![image](https://github.com/priyanshu1947/Stroke_Prediction/assets/70458921/3020ebbd-7148-4434-8c79-e77d83243089)


- Use one-hot encoding or other encoding techniques to convert categorical features into numerical.

## Separate Dependent and Independent Features
![image](https://github.com/priyanshu1947/Stroke_Prediction/assets/70458921/82c97c44-8a56-4660-9264-a51ce5d78f23)

- Identify and separate the target variable (dependent feature) from the rest of the dataset (independent features).

## Scale the Data
![image](https://github.com/priyanshu1947/Stroke_Prediction/assets/70458921/4f67fe54-a63f-4c2a-9fec-3247084f333c)

- Use `StandardScaler` to scale the numerical features.

## Split the Dataset
![image](https://github.com/priyanshu1947/Stroke_Prediction/assets/70458921/50a59e7f-5029-4460-bd12-dfae670c40aa)


- Split the dataset into training and testing sets using `train_test_split`.

## Building Classifier

- Initialize and train machine learning models for classification.

## Logistic Regression
![image](https://github.com/priyanshu1947/Stroke_Prediction/assets/70458921/85b9e03b-b8eb-478d-82ff-ebbceb25c9a2)


- Train a logistic regression model using `LogisticRegression`.

## Confusion Matrix
![image](https://github.com/priyanshu1947/Stroke_Prediction/assets/70458921/a0d53472-fe79-46da-a23a-61b8d3ce583a)


- Evaluate the performance of the logistic regression model using a confusion matrix.

## Handling Imbalanced Data using SMOTE
![image](https://github.com/priyanshu1947/Stroke_Prediction/assets/70458921/9e588921-8acb-4123-b690-fea1b16dc7ff)

- Use Synthetic Minority Over-sampling Technique (SMOTE) to handle imbalanced data.

## Splitting the Oversampling Data
![image](https://github.com/priyanshu1947/Stroke_Prediction/assets/70458921/abe4f2f0-0e56-41c8-a026-e1b08c246902)

- Split the oversampled data into training and testing sets.

## Using KNN
![image](https://github.com/priyanshu1947/Stroke_Prediction/assets/70458921/62a977da-3bcc-4afb-8ee1-6a2c5520760f)


- Train a K-Nearest Neighbors classifier using `KNeighborsClassifier`.

## Using TensorFlow Keras

- Build and train a neural network using `tensorflow.keras.Sequential`.
  ![image](https://github.com/priyanshu1947/Stroke_Prediction/assets/70458921/6478e4a3-2621-4873-a823-4218f7e7bccc)
  ![image](https://github.com/priyanshu1947/Stroke_Prediction/assets/70458921/27bb51d2-7352-4cc6-bf28-8ac2954b5310)
  ![image](https://github.com/priyanshu1947/Stroke_Prediction/assets/70458921/70480b85-8330-4633-af1f-5f76d3bbc297)
  ![image](https://github.com/priyanshu1947/Stroke_Prediction/assets/70458921/c27f838c-d89f-49a2-88e9-53f7dc1600f0)

## Installation

 Clone the repository:

   ```bash
   git clone https://github.com/your-username/stroke-prediction.git
   cd stroke-prediction

## Future Work

- Suggest potential improvements or additional analyses.
- Consider incorporating more advanced algorithms or fine-tuning hyperparameters.



