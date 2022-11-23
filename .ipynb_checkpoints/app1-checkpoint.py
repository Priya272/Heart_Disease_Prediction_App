import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns

df=pd.read_csv('Heart_Disease_Prediction.csv')

st.title('Heart Disease Predictor')
st.sidebar.header('Patient Data')
st.subheader('Training Data Stats')
st.write(df.describe())

x=df.drop(['Heart Disease'],axis=1)
y=df.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)

def user_report():
  age = st.sidebar.slider('Age', 29,77, 54 )
  cholesterol= st.sidebar.slider('Cholesterol',126,564,250)
  bp = st.sidebar.slider('Blood Pressure', 94,200, 132 )
  stdepression= st.sidebar.slider('ST depression', 0, 6.2, 1.5 )
  chestpain = st.sidebar.slider('Chest pain type', 1,4, 0.95 )
  maxhr = st.sidebar.slider('Max HR', 71,202, 150 )
  thallium = st.sidebar.slider('Thallium', 3,7, 4.7 )
  exerciseangina = st.sidebar.slider('Exercise angina', 0,1, 0.5 )

 #created a dictionery

  user_report_data = {
      'age':age,
      'cholesterol':cholesterol,
      'bp':bp,
      'stdepression':stdepression,
      'chestpain':chestpain,
      'maxhr':maxhr,
      'thallium':thallium,
      'exerciseangina':exerciseangina
  }

  #imported the dictionery into a dataframe

  report_data = pd.DataFrame(user_report_data, index=[0])
  return report_data


user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)

# MODEL
rf  = RandomForestClassifier()
rf.fit(x_train, y_train)
user_result = rf.predict(user_data)



# VISUALISATIONS
#st.title('Visualised Patient Report')




# OUTPUT
st.subheader('Your Report: ')
output=''
if user_result[0]==0:
  output = 'You are not Diabetic'
else:
  output = 'You are Diabetic'
st.title(output)
st.subheader('Accuracy: ')
st.write(str(accuracy_score(y_test, rf.predict(x_test))*100)+'%')