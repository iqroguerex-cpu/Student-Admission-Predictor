import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# ---------------------------
# Load Dataset
# ---------------------------
dataset = pd.read_csv("Student_Admission.csv")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# ---------------------------
# Handle Missing Values
# ---------------------------
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(X[:, 2:4])
X[:, 2:4] = imputer.transform(X[:, 2:4])

# ---------------------------
# Encode Categorical Features
# ---------------------------
ct = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(), [0, 1, 4])],
    remainder="passthrough"
)

X = np.array(ct.fit_transform(X))

# Encode Target
le = LabelEncoder()
y = le.fit_transform(y)

# ---------------------------
# Split Data
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# ---------------------------
# Feature Scaling
# ---------------------------
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# ---------------------------
# Train Model
# ---------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# ---------------------------
# Streamlit UI
# ---------------------------

st.title("🎓 Student Admission Predictor")

st.sidebar.header("Enter Student Details")

country = st.sidebar.selectbox("Country", ["India", "USA", "Germany"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
gre = st.sidebar.slider("GRE Score", 280, 350, 320)
cgpa = st.sidebar.slider("CGPA", 6.0, 10.0, 8.0)
research = st.sidebar.selectbox("Research Experience", ["Yes", "No"])

# Convert user input to dataframe
input_df = pd.DataFrame(
    [[country, gender, gre, cgpa, research]],
    columns=["Country", "Gender", "GRE_Score", "CGPA", "Research"]
)

# Apply preprocessing to input
input_X = ct.transform(input_df)
input_X = sc.transform(input_X)

# Prediction
prediction = model.predict(input_X)

if st.button("Predict Admission"):
    if prediction[0] == 1:
        st.success("✅ Student is likely to be ADMITTED")
    else:
        st.error("❌ Student is NOT likely to be admitted")

# ---------------------------
# Charts Section
# ---------------------------
st.subheader("📊 Dataset Insights")

# Chart 1: GRE Distribution
fig1 = plt.figure()
plt.hist(dataset["GRE_Score"].dropna(), bins=10)
plt.title("GRE Score Distribution")
plt.xlabel("GRE Score")
plt.ylabel("Frequency")
st.pyplot(fig1)

# Chart 2: Admission Count
fig2 = plt.figure()
dataset["Admitted"].value_counts().plot(kind="bar")
plt.title("Admission Count")
st.pyplot(fig2)
