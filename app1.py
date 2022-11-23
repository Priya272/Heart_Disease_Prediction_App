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
Outcome = pd.get_dummies(df['Heart Disease'])
df = df.drop('Heart Disease',axis = 1)
df = df.join(Outcome)
df=df.drop('Absence',axis=1)
df.rename(columns={"Presence":"Outcome"},inplace='True')

x=df.drop(['Outcome'],axis=1)
y=df.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)

def user_report():
  age = st.sidebar.slider('Age', 29,77, 54 )
  cholesterol= st.sidebar.slider('Cholesterol',126,564,250)
  bp = st.sidebar.slider('BP', 94,200, 132 )
  stdepression= st.sidebar.slider('ST depression', 0, 6, 1 )
  chestpain = st.sidebar.slider('Chest pain type', 1,4, 1 )
  maxhr = st.sidebar.slider('Max HR', 71,202, 150 )
  thallium = st.sidebar.slider('Thallium', 3,7, 5 )
  sex = st.sidebar.slider('Sex', 0,1, 0 )
  exerciseangina = st.sidebar.slider('Exercise angina', 0,1, 0 )
  fbs = st.sidebar.slider('FBS over 120', 0,1, 0 )
  slopeofst= st.sidebar.slider('Slope of ST', 1,3, 1 )
  novf = st.sidebar.slider('Number of vessels fluro', 0,3, 1 )
  ekgresult = st.sidebar.slider('EKG results', 0,2, 1 )



  user_report_data = {
      'age':age,
      'cholesterol':cholesterol,
      'bp':bp,
      'stdepression':stdepression,
      'chestpain':chestpain,
      'maxhr':maxhr,
      'thallium':thallium,
      'exerciseangina':exerciseangina,
      'sex':sex,
      'fbs':fbs,
      'slopeofst':slopeofst,
      'novf':novf,
      'ekgresult':ekgresult
    
  }

  #imported the dictionery into a dataframe

  report_data = pd.DataFrame(user_report_data, index=[0])
  return report_data


user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)

# MODEL
rf  = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
rf.fit(x_train, y_train)
user_result = rf.predict(user_data)

# VISUALISATIONS
st.title('Visualised Patient Report')


# COLOR FUNCTION
if user_result[0]==0:
  color = 'blue'
else:
  color = 'red'


# Age vs bp
st.header('Blood Pressure count Graph (Others vs Yours)')
fig_bp = plt.figure()
ax1 = sns.scatterplot(x = 'Age', y = 'BP', data = df, hue = 'Outcome', palette = 'PuBu')
ax2 = sns.scatterplot(x = user_data['age'], y = user_data['bp'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,250,10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_bp)


# Age vs Cholesterol
st.header('Cholesterol count Graph (Others vs Yours)')
fig_chole = plt.figure()
ax3 = sns.scatterplot(x = 'Age', y = 'Cholesterol', data = df, hue = 'Outcome' , palette='magma')
ax4 = sns.scatterplot(x = user_data['age'], y = user_data['cholesterol'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,600,100))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_chole)


# Age vs Chest pain type
st.header('Chest Pain type Value Graph (Others vs Yours)')
fig_chestp = plt.figure()
ax5 = sns.scatterplot(x = 'Age', y = 'Chest pain type', data = df, hue = 'Outcome', palette='RdGy')
ax6 = sns.scatterplot(x = user_data['age'], y = user_data['chestpain'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(1,4,1))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_chestp)


# Age vs Max Hr
st.header('Max HR Value Graph (Others vs Yours)')
fig_maxhr = plt.figure()
ax7 = sns.scatterplot(x = 'Age', y = 'Max HR', data = df, hue = 'Outcome', palette='Dark2')
ax8 = sns.scatterplot(x = user_data['age'], y = user_data['maxhr'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(50,210,10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_maxhr)


# Age vs EKG results
st.header('EKG results Value Graph (Others vs Yours)')
fig_ekg = plt.figure()
ax9 = sns.scatterplot(x = 'Age', y = 'EKG results', data = df, hue = 'Outcome', palette='summer_r')
ax10 = sns.scatterplot(x = user_data['age'], y = user_data['ekgresult'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,900,50))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_ekg)


# OUTPUT
st.subheader('Your Report: ')
output=''
if user_result[0]==0:
  output = 'You do not have any Heart Disease'
else:
  output = 'You Have Heart Disease'
st.title(output)
st.subheader('Accuracy: ')
st.write(str(accuracy_score(y_test, rf.predict(x_test))*100)+'%')