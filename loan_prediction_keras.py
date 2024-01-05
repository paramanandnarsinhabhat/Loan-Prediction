# Pre-processing Steps
'''
1. Filling the missing values
2. Converting categories to numbers
3. Bring all the variables in range 0 to 1
'''
# importing required libraries
import pandas as pd

# check version on pandas
print('Version of pandas:', pd.__version__)

# reading the loan prediction data
data = pd.read_csv('/Users/paramanandbhat/Downloads/Loan Prediction pre-processing/loan_data.csv')

# looking at the first five rows of the data
data.head()

print(data.head())

# shape of the data
data.shape


print(data.shape)

# checking missing values in the data
data.isnull().sum()

print(data.isnull().sum())

# data types of the variables
data.dtypes

print(data.dtypes)

## 1. Filling the missing values
# filling missing values of categorical variables with mode

data['Gender'].fillna(data['Gender'].mode()[0], inplace=True)

data['Married'].fillna(data['Married'].mode()[0], inplace=True)

data['Dependents'].fillna(data['Dependents'].mode()[0], inplace=True)

data['Self_Employed'].fillna(data['Self_Employed'].mode()[0], inplace=True)

data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0], inplace=True)

data['Credit_History'].fillna(data['Credit_History'].mode()[0], inplace=True)

### Continuous Data: Mean
# filling missing values of continuous variables with mean
data['LoanAmount'].fillna(data['LoanAmount'].mean(), inplace=True)

# checking missing values after imputation
data.isnull().sum()

## 2. Converting categories to numbers
# converting the categories into numbers using map function
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})
data['Married'] = data['Married'].map({'No': 0, 'Yes': 1})
data['Dependents'] = data['Dependents'].map({'0': 0, '1': 1, '2': 2, '3+': 3})
data['Education'] = data['Education'].map({'Graduate': 1, 'Not Graduate': 0})
data['Self_Employed'] = data['Self_Employed'].map({'No': 0, 'Yes': 1})
data['Property_Area'] = data['Property_Area'].map({'Rural': 0, 'Semiurban': 1, 'Urban': 2})
data['Loan_Status'] = data['Loan_Status'].map({'N': 0, 'Y': 1})

print(data.head())





