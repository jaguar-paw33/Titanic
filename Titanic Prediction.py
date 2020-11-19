#!/usr/bin/env python
# coding: utf-8

#Loading Libraries
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix 
from sklearn import preprocessing 


# Dataset Description
st.markdown(body='''
<h1 style='color:#024661; text-align:center;'>Titanic Survival Prediction</h1>
<br>
<style>
body {
background-image: url("https://cutewallpaper.org/21/gradient-hd-background/Blue-Gradient-Background-4K-HD-Desktop-Wallpaper-for-4K-.jpg");
background-size: cover;
background-position:center;
background-attachment: fixed;
color:#1f414f;
}
</style>

<br>
<img style="border-radius:6px;" src='https://static.toiimg.com/photo/58787332.cms' width=96%>
<br>
<br>

<p>Titanic Survival Dataset consists data about 712 passengers who were\
 there on the ship, each passenger is described by the following characterstics,\
   <ol>
    <li>Survived
      <ul>
        <li>No - 0</li>
        <li>Yes - 1</li>
      </ul>
    </li>
    <li>Passenger Class (Pclass)
      <ul>
        <li>First - 1</li>
        <li>Second - 2</li>
        <li>Third - 3</li>
      </ul>
    </li>
    <li>Sex
      <ul>
        <li>Female - 0</li>
        <li>Male - 1</li>
      </ul>
    </li>
    <li>Age</li>
    <li>Number of Siblings/Spouses Aboard (SibSp)</li>
    <li>Number of Parents/Children Aboard (Parch)</li>
    <li>Passenger Fare (British Pound)</li>
    
   </ol>
   <br>
   <div style='border-bottom:2px Solid Black'></div>
   <br>
''', unsafe_allow_html=True)

#Loading Data
@st.cache(allow_output_mutation=True)
def load_data():
  data=pd.read_csv('train.csv')
  return data


st.header('Loading Dataset')
data_load_state=st.text('Loading Data...')
titanic=load_data()
data_load_state.text('Data Loaded Successfully!!')

# Data Preprocessing 
titanic = titanic.dropna(subset =['Embarked', 'Age'])
titanic=titanic.drop(['PassengerId','Name','Ticket','Cabin','Embarked' ], axis=1)
label_encoder = preprocessing.LabelEncoder() 
titanic['Sex']= label_encoder.fit_transform(titanic['Sex']) 
titanic['Age']=titanic['Age'].astype('int64')
titanic=titanic.reset_index(drop=True)

#Show Dataset
if st.checkbox('Show Dataset'):
  st.subheader('Titanic Survival Dataset')
  st.write(titanic)
  st.write('Dimensionality of the Dataset')
  st.write(titanic.shape)
  st.write("Description of the Dataset")
  st.write(titanic.describe())
  st.write("Number of Survived(1) and Died(0) passengers")
  st.write(titanic['Survived'].value_counts())

  #Exploratory Data Analysis
  st.markdown(body='''
  <br>
  <h4>Exploratory Data Analysis</h4>
  <br>
  ''', unsafe_allow_html=True)

  if(st.checkbox('Show Plots')):
    custom_pallete=['#13a9d6','#6cd5f5','#a3d3e3']
    sns.set_style("darkgrid")
    sns.set_context("paper")
    sns.despine()
    sns.set_palette(custom_pallete)

    fig1,ax1=plt.subplots()
    ax1.set_title('Count Plot for Survival Status')
    ax1=sns.countplot(titanic['Survived'])
    st.pyplot(fig1)
    st.markdown(body='''
      <strong>Observation(s):-</strong> <span style="font">Number of passengers survived are lesser than the number of passengers died.</span>
      <br>
      <br>
    ''', unsafe_allow_html=True)


    fig7,ax7=plt.subplots()
    ax7.set_title('Survival Status vs Male/Female')
    ax7 = sns.countplot(x="Survived", hue="Sex", data=titanic)
    st.pyplot(fig7)
    st.markdown(body='''
      <strong>Observation(s):-</strong> <span style="font">
        <ul>
        <li>Passengers who survived consists of more female passengers than male passengers.</li>
        <li>Passengers who died had larger number of males as compared to females.</li>
        </ul>
      </span>
      <br>
      <br>
    ''', unsafe_allow_html=True)


    fig5,ax5=plt.subplots()
    ax5.set_title('Count Plot for Passenger Class')
    ax5=sns.countplot(x="Pclass", data=titanic)
    st.pyplot(fig5)
    st.markdown(body='''
      <strong>Observation(s):-</strong>
      <ul>
        <li>Highest number of passangers belonged to Class 3.</li>
        <li>Class 1 and Class 2 had almost same number of passengers.</li>
      </ul>
      </span>
      <br>
      <br>
    ''', unsafe_allow_html=True)

    fig6,ax6=plt.subplots()
    ax6.set_title('Class vs Male/Female')
    ax6 = sns.countplot(x="Pclass", hue="Sex", data=titanic)
    st.pyplot(fig6)
    st.markdown(body='''
      <strong>Observation(s):-</strong> <span style="font">
        <ul>
        <li>Class 3 had a comparatively larger number of male passengers than female passengers.</li>
        <li>The difference in the number of male and female passengers was not much pronounced in Class 1 and Class 2.</li>
        </ul>
      </span>
      <br>
      <br>
    ''', unsafe_allow_html=True)

    fig2,ax2=plt.subplots()
    ax2=sns.catplot(data=titanic, kind="swarm", x="Pclass", y="Age", hue="Survived") 
    st.pyplot(ax2)
    st.markdown(body='''
    <strong>Observation(s):-</strong> <span style="font">
      <ul>
      <li>Class 3 had the highest number of young passengers followed by Class 2 and then Class 1.</li>
      <li>Class 1 had most of the old passengers followed by Class 2 and then Class 3.</li>
      </ul>
    </span>    
    <br>
    <br>
  ''', unsafe_allow_html=True)

    fig3,ax3=plt.subplots()
    ax3.set_title('Class vs Chance of Survival')
    ax3=sns.barplot(x='Pclass', y='Survived', data=titanic)
    st.pyplot(fig3)
    st.markdown(body='''
      <strong>Observation(s):-</strong> <span style="font">The chances of survival were highest for people of higher class and the chances reduced as the class Reduced.</span>
      <br>
      <br>
    ''', unsafe_allow_html=True)


    fig4,ax4=plt.subplots()
    ax4.set_title('Fair vs Class')
    ax4=sns.boxplot(x='Pclass', y='Fare', data=titanic)
    st.pyplot(fig4)
    st.markdown(body='''
      <strong>Observation(s):-</strong> <span style="font">Fair was highest for Class 1 and comparatively lesser for Class 2 and least for Class 3.</span>
      <br>
      <br>
    ''', unsafe_allow_html=True)


st.write('\n')


#Training Data
X=titanic
Y=X['Survived']
X=X.drop(['Survived'], axis=1)


# Split the dataset into 80% Training set and 20% Testing set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


#Function to train differnt Models
@st.cache(allow_output_mutation=True)
def models(X_train,Y_train):
  #Using Logistic Regression Algorithm to the Training Set
  from sklearn.linear_model import LogisticRegression
  log = LogisticRegression(random_state=3)
  log.fit(X_train, Y_train)
  #Using KNeighborsClassifier Method of neighbors class to use Nearest Neighbor algorithm
  from sklearn.neighbors import KNeighborsClassifier
  knn = KNeighborsClassifier(n_neighbors = 5)
  knn.fit(X_train, Y_train)

  #Using SVC method of svm class to use Support Vector Machine Algorithm
  from sklearn.svm import SVC
  svc_lin = SVC(kernel = 'linear',random_state=3)
  svc_lin.fit(X_train, Y_train)

  #Using SVC method of svm class to use Kernel SVM Algorithm
  from sklearn.svm import SVC
  svc_rbf = SVC(kernel = 'rbf',random_state=3)
  svc_rbf.fit(X_train, Y_train)

  #Using GaussianNB method of naïve_bayes class to use Naïve Bayes Algorithm
  from sklearn.naive_bayes import GaussianNB
  gauss = GaussianNB()
  gauss.fit(X_train, Y_train)

  #Using DecisionTreeClassifier of tree class to use Decision Tree Algorithm
  from sklearn.tree import DecisionTreeClassifier
  tree = DecisionTreeClassifier(criterion = 'entropy',random_state=3)
  tree.fit(X_train, Y_train)

  #Using RandomForestClassifier method of ensemble class to use Random Forest Classification algorithm
  from sklearn.ensemble import RandomForestClassifier
  forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state=3)
  forest.fit(X_train, Y_train)
  
  return log, knn, svc_lin, svc_rbf, gauss, tree, forest


#Function to print the Confusion Matrix
def confusionMatrix(model, name):
    cm = confusion_matrix(Y_test, model.predict(X_test)) 
    #extracting TN, FP, FN, TP
    TN, FP, FN, TP = confusion_matrix(Y_test, model.predict(X_test)).ravel()
    st.write("Confusion Matrix\n",cm)
    st.write(name, "Testing Accuracy = ", (TP + TN) / (TP + TN + FN + FP))


#Function to Make Prediction
def make_prediction(query,model):
  predict=model.predict(query)
  if(predict):
    st.markdown(body='''
      <br>
    	<div style='font-size:100px; text-align:center;'>&#128512;</div>	
      <div style='text-align:center;'><h2>Congratulations!! You would have Survived.</h2></div>
    ''', unsafe_allow_html=True)
  else:
     st.markdown(body='''
      <br>
    	<div style='font-size:100px; text-align:center;'>&#128532;</div>	
      <div style='text-align:center;'><h2>Sorry!! You Wouldn't have Survived.</h2></div>
    ''', unsafe_allow_html=True)

st.markdown(body='''
   <div style='border-bottom:2px Solid Black'></div>
   <br>
''', unsafe_allow_html=True)

#Training Models
st.header('Trainig Models')
data_load_state=st.text('Training in Progress...')
log,knn,svc_lin,svc_rbf,gauss,tree,forest=models(X_train, Y_train)
data_load_state.text('Models Trained Successfully!!')


#Show Performance of Models after Training
if(st.checkbox('Show Performance of Models')):
  st.subheader('Performance of Various Models on Same Dataset-')
  st.markdown(body="<br>",unsafe_allow_html=True)
  
  st.write('1.Logistic Regression Training Accuracy:', log.score(X_train, Y_train))
  confusionMatrix(log, "Logistic Regression")
  st.markdown(body="<br>",unsafe_allow_html=True)

  st.write('2. K Nearest Neighbor Training Accuracy:', knn.score(X_train, Y_train))
  confusionMatrix(knn,"K Nearest Neighbor")
  st.markdown(body="<br>",unsafe_allow_html=True)
  
  st.write('3. Support Vector Machine (Linear Classifier) Training Accuracy:', svc_lin.score(X_train, Y_train))
  confusionMatrix(svc_lin,"Support Vector Machine (Linear Classifier)")
  st.markdown(body="<br>",unsafe_allow_html=True)
  
  st.write('4. Support Vector Machine (RBF Classifier) Training Accuracy:', svc_rbf.score(X_train, Y_train))
  confusionMatrix(svc_rbf,"Support Vector Machine (RBF Classifier)")
  st.markdown(body="<br>",unsafe_allow_html=True)
  
  st.write('5. Gaussian Naive Bayes Training Accuracy:', gauss.score(X_train, Y_train))
  confusionMatrix(gauss,"Gaussian Naive Bayes")
  st.markdown(body="<br>",unsafe_allow_html=True)
  
  st.write('6. Decision Tree Classifier Training Accuracy:', tree.score(X_train, Y_train))
  confusionMatrix(tree,"Decision Tree Classifier")
  st.markdown(body="<br>",unsafe_allow_html=True)
  
  st.write('7. Random Forest Classifier Training Accuracy:', forest.score(X_train, Y_train))
  confusionMatrix(forest,"Random Forest Classifier")
  st.markdown(body='''
  <br> 
  <h3><em>Random Forest is performing best for this dataset.</em></h3><br><br><br>
  ''', unsafe_allow_html=True)

if(st.checkbox('Show Importance of features in Predicting the Survival Status of Passengers')):
  importances = pd.DataFrame({'feature':titanic.iloc[:, 1:8].columns,'Importance':np.round(forest.feature_importances_,3)})
  importances = importances.sort_values('Importance',ascending=False).set_index('feature')
  importances

st.markdown(body='''
  <br>
  <div style='border-bottom:2px Solid Black'></div>
''', unsafe_allow_html=True)

#Making Predictions 
st.write('\n')
st.header('Know your Survival Status (if you were on the Titanic)')
cl=st.number_input('Class',value=1, min_value=1, max_value=3)
sex=st.number_input('Sex',value=0, min_value=0, max_value=1)
age=st.number_input('Age',value=1, min_value=1, max_value=150)
sibsp=st.number_input('Number of Siblings/Spouses Aboard',value=0,min_value=0, max_value=100)
parch=st.number_input('Number of Parents/Children Aboard',value=0,min_value=0, max_value=100)
fare=st.number_input('Fare',value=5,min_value=5, max_value=200)

query=[[cl,sex,age,sibsp,parch,fare]]


if(st.button('Make Prediction')):
  make_prediction(query,forest)









