import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from collections import Counter
import streamlit as st

# Function to embed HTML code for linking CSS file
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Call the local_css function and provide the name of your CSS file
local_css("stream.css")


# Load data
df = pd.read_csv("bank.csv")

# Hide warnings
st.set_option('deprecation.showPyplotGlobalUse', False)


# Show data description
st.subheader("Data Description")
st.write(df.describe())

# Show missing values
st.subheader("Missing Values")
st.write(df.isnull().sum())

# Visualizations
st.subheader("Visualizations")

# Distribution plot of balance
st.write("Distribution of balance")
sns.distplot(df['balance'])
st.pyplot()

# Skewness and kurtosis of balance
st.write("Skewness: ", df['balance'].skew())
st.write("Kurtosis: ", df['balance'].kurt())

# Distribution plot of age
st.write("Distribution of age")
sns.distplot(df['age'])
st.pyplot()

# Skewness and kurtosis of age
st.write("Skewness: ", df['age'].skew())
st.write("Kurtosis: ", df['age'].kurt())

# Distribution plot of duration
st.write("Distribution of duration")
sns.distplot(df['duration'])
st.pyplot()

# Count plot of marital status
st.write("Count of different marital status")
sns.countplot(x=df['marital'])
st.pyplot()

# Count plot of job category
st.write("Count of different job category")
sns.countplot(x=df['job'], hue=df['deposit'])
st.pyplot()

# Distribution plot of campaign
st.write("Count of Campaign made")
sns.distplot(df['campaign'], kde=False)
st.pyplot()

# Count plot of target variable
st.write("Count of target variable that is subscription")
sns.countplot(x=df['deposit'])
st.pyplot()

# Boxplot of age
st.write("Boxplot for Age Feature")
sns.boxplot(x=df['age'])
st.pyplot()

# Boxplot of duration
st.write("Boxplot for Duration Feature")
sns.boxplot(x=df['duration'])
st.pyplot()

# Boxplot of campaign
st.write("Boxplot for Campaign Feature")
sns.boxplot(x=df['campaign'])
st.pyplot()

# Remove outliers
Q1 = df['balance'].quantile(0.25)
Q3 = df['balance'].quantile(0.75)
IQR = Q3 - Q1
lower_lim = Q1 - 1.5 * IQR
upper_lim = Q3 + 1.5 * IQR
df.loc[df['balance'] > 4087.0, "balance"] = 4087
df.loc[df['age'] > 74, "age"] = 74
df.loc[df['duration'] > 1033, "duration"] = 1033
df = df[(df['campaign'] > lower_lim) & (df['campaign'] < upper_lim)]

# Box plot after removing outliers
st.write("Box plot of Duration after removing outliers")
sns.boxplot(x=df['duration'])
st.pyplot()

st.write("Box plot of balance after removing outliers")
sns.boxplot(x=df['balance'])
st.pyplot()

st.write("Box plot of age after removing outliers")
sns.boxplot(x=df['age'])
st.pyplot()

st.write("Box plot of campaign after removing outliers")
sns.boxplot(x=df['campaign'])
st.pyplot()

# Analysing balance features after removing outliers
st.write("Distribution of balance after removing outliers")
sns.distplot(df['balance'])
st.pyplot()

# Analysing duration features after removing outliers
st.write("Distribution of duration after removing outliers")
sns.distplot(df['duration'])
st.pyplot()

# Print count of each value in particular feature columns
st.write("Value Counts after Removing Unknowns")
st.write("poutcome", df['poutcome'].value_counts())
st.write("contact", df['contact'].value_counts())
st.write("education", df['education'].value_counts())
st.write("job", df['job'].value_counts())

# Replace unknown values
df['poutcome'] = df['poutcome'].replace(['unknown'], 'failure')
df['contact'] = df['contact'].replace(['unknown'], 'cellular')
df['education'] = df['education'].replace(['unknown'], 'secondary')
df['job'] = df['job'].replace(['unknown'], 'management')

# Print the count of values after filling the unknown values.
st.write("Value Counts after Filling Unknowns")
st.write("poutcome", df['poutcome'].value_counts())
st.write("contact", df['contact'].value_counts())
st.write("education", df['education'].value_counts())
st.write("job", df['job'].value_counts())

# Heatmap plot to find which two variables are correlated
numeric_df = df.select_dtypes(include=['int64', 'float64'])
plt.figure(figsize=(15, 10))
sns.heatmap(numeric_df.corr(), annot=True)
st.write("Heatmap to show correlation between numerical variables")
st.pyplot()

# Encoding categorical feature
df['default'].replace({'yes': 1, 'no': 0}, inplace=True)
df['housing'].replace({'yes': 1, 'no': 0}, inplace=True)
df['loan'].replace({'yes': 1, 'no': 0}, inplace=True)
df['deposit'].replace({'yes': 1, 'no': 0}, inplace=True)
df['contact'].replace({'cellular': 1, 'telephone': 0}, inplace=True)
df['month'].replace({'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9,
                     'oct': 10, 'nov': 11, 'dec': 12}, inplace=True)

# One hot encoding for other categorical features
data1 = pd.get_dummies(data=df, columns=['poutcome', 'education', 'marital', 'job'])

# Perform standard scaler to scale down the values
scaled_col = ['age', 'balance', 'duration', 'pdays']
scaler = StandardScaler()
data1[scaled_col] = scaler.fit_transform(data1[scaled_col])

# Split the data into train and test (75 : 25)
x = data1.drop(['deposit'], axis=1)
y = data1['deposit']  # target feature
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Apply SMOTETomek to balance classes
smk = SMOTETomek()
X_res, y_res = smk.fit_resample(x_train, y_train)

# Plotting count of target variable after balancing classes
st.write("Count of target variable after balancing classes")
sns.countplot(x=y_res, data=data1)
plt.xticks(fontsize=12, rotation=0)
plt.yticks(fontsize=12)
st.pyplot()

# Training model with Logistic regression
lg = LogisticRegression()
lg.fit(X_res, y_res)
Lg_pred = lg.predict(x_test)

# Get the accuracy and classification report
st.write("Logistic Regression Model Performance")
st.write("Accuracy: ", metrics.accuracy_score(y_test, Lg_pred))
st.write("Confusion Matrix:")
st.write(metrics.confusion_matrix(y_test, Lg_pred))
st.write("Classification Report:")
st.write(metrics.classification_report(y_test, Lg_pred))

# Building model with Random Forest
rf = RandomForestClassifier()
rf.fit(X_res, y_res)
rf_pred = rf.predict(x_test)

st.write("Random Forest Model Performance")
st.write("Accuracy: ", metrics.accuracy_score(y_test, rf_pred))
st.write("Confusion Matrix:")
st.write(metrics.confusion_matrix(y_test, rf_pred))
st.write("Classification Report:")
st.write(metrics.classification_report(y_test, rf_pred))

# Linear SVC
lr_svc = SVC(kernel='rbf')
lr_svc.fit(X_res, y_res)
sv_pred = lr_svc.predict(x_test)

st.write("Linear SVC Model Performance")
st.write("Accuracy: ", metrics.accuracy_score(y_test, sv_pred))
st.write("Confusion Matrix:")
st.write(metrics.confusion_matrix(y_test, sv_pred))
st.write("Classification Report:")
st.write(metrics.classification_report(y_test, sv_pred))

# XGBoost
xg = XGBClassifier()
xg.fit(X_res, y_res)
xg_pred = xg.predict(x_test)

st.write("XGBoost Model Performance")
st.write("Accuracy: ", metrics.accuracy_score(y_test, xg_pred))
st.write("Confusion Matrix:")
st.write(metrics.confusion_matrix(y_test, xg_pred))
st.write("Classification Report:")
st.write(metrics.classification_report(y_test, xg_pred))

# Build model with KNN
knn = KNeighborsClassifier()
knn.fit(X_res, y_res)
knn_pred = knn.predict(x_test)

st.write("KNN Model Performance")
st.write("Accuracy: ", metrics.accuracy_score(y_test, knn_pred))
st.write("Confusion Matrix:")
st.write(metrics.confusion_matrix(y_test, knn_pred))
st.write("Classification Report:")
st.write(metrics.classification_report(y_test, knn_pred))

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_res, y_res)
dt_pred = dt.predict(x_test)

st.write("Decision Tree Model Performance")
st.write("Accuracy: ", metrics.accuracy_score(y_test, dt_pred))
st.write("Confusion Matrix:")
st.write(metrics.confusion_matrix(y_test, dt_pred))
st.write("Classification Report:")
st.write(metrics.classification_report(y_test, dt_pred))
