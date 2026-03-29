%%writefile app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


st.title("📊 Telecom Churn Dashboard")

# Load data
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Convert churn
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Preprocessing
df.drop('customerID', axis=1, inplace=True)

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Fix NaN issue
df.dropna(subset=['Churn'], inplace=True)

# Encode
df = pd.get_dummies(df, drop_first=True)

# Show data
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Churn distribution
st.subheader("Churn Distribution")
st.bar_chart(df['Churn'].value_counts())

# Split
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Monthly Charges vs Churn
st.subheader("Monthly Charges Distribution")
fig, ax = plt.subplots()
df[df['Churn']==1]['MonthlyCharges'].hist(alpha=0.7)
df[df['Churn']==0]['MonthlyCharges'].hist(alpha=0.7)
ax.legend(['Churn', 'No Churn'])
st.pyplot(fig)

# Tenure vs Churn
st.subheader("Tenure Distribution")
fig, ax = plt.subplots()
df[df['Churn']==1]['tenure'].hist(alpha=0.7)
df[df['Churn']==0]['tenure'].hist(alpha=0.7)
ax.legend(['Churn', 'No Churn'])
st.pyplot(fig)

# ================= UI =================

st.subheader("📌 Model Performance")

accuracy = accuracy_score(y_test, y_pred)
st.write(f"Accuracy: {accuracy:.2f}")

# Confusion Matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix')
st.pyplot(fig)

# ================= Feature Importance =================

st.subheader("🔥 Top 10 Important Features")

feature_importances = pd.Series(model.feature_importances_, index=X.columns)
top_features = feature_importances.nlargest(10)

fig, ax = plt.subplots()
top_features.sort_values().plot(kind='barh', ax=ax)
st.pyplot(fig)


