import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

data=pd.read_csv(r"C:\AYUSH N P\data science course\Machine learning\Iris-train.csv")
print(data.head())

x = data.iloc[:,1:-1]
print(x.head())
y = data.iloc[:,-1]

model=RandomForestClassifier()
model.fit(x,y)


st.title("Iris Machine Learning Classifier")
st.subheader("Enter your inputs here in the form")


with st.form(key='iris_form'):
    sepal_length = st.slider("Enter the Sepal Length (cm):", 4.3, 7.9, 5.8, step=0.1)
    sepal_width = st.slider("Enter the Sepal Width (cm):", 2.0, 4.4, 3.0, step=0.1)
    petal_length = st.slider("Enter the Petal Length (cm):", 1.0, 6.9, 4.3, step=0.1)
    petal_width = st.slider("Enter the Petal Width (cm):", 0.1, 2.5, 1.3, step=0.1)

    if st.form_submit_button():
        st.success("Your Iris Machine Learning Classifier is loaded")
        prediction=model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
        st.subheader(f"It is {prediction[0]}")
