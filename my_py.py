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
st.title("Titanic Survival Prediction")
st.markdown(body='''
<br>
<img src='https://upload.wikimedia.org/wikipedia/commons/9/9c/Titanic_wreck_bow.jpg' height='400' width='670'>
<br>
<br>

<p>Titanic Survival Dataset consists data about 848 passengers who were\
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

#Show Dataset
if st.checkbox('Show Dataset'):
  st.subheader('Titanic Survival Dataset')
  st.write(titanic)
  st.write('Dimensions of the Dataset')
  st.write(titanic.shape)
  st.write("Description of Dataset")
  st.write(titanic.describe())
  st.write("Number of Survived(1) and Died(0) people in the dataset")
  st.write(titanic['Survived'].value_counts())

  #Exploratory Data Analysis
  st.markdown(body='''
  <br>
  <h4>Exploratory Data Analysis</h4>

  ''', unsafe_allow_html=True)

  if(st.checkbox('Show Plots')):
    fig1,ax1=plt.subplots()
    ax1.set_title('Count Plot')
    sns.countplot(titanic['Survived'])
    st.pyplot(fig1)
    
    fig2,ax2=plt.subplots()
    ax2.set_title('Bar Plot')
    sns.barplot(x='Pclass', y='Survived', data=titanic)
    st.pyplot(fig2)
    
    fig3,ax3=plt.subplots()
    ax3.set_title('Fair vs Class')
    sns.boxplot(x='Pclass', y='Fare', data=titanic)
    st.pyplot(fig3)


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
  log = LogisticRegression(random_state=0)
  log.fit(X_train, Y_train)
  
  #Using KNeighborsClassifier Method of neighbors class to use Nearest Neighbor algorithm
  from sklearn.neighbors import KNeighborsClassifier
  knn = KNeighborsClassifier(n_neighbors = 5)
  knn.fit(X_train, Y_train)

  #Using SVC method of svm class to use Support Vector Machine Algorithm
  from sklearn.svm import SVC
  svc_lin = SVC(kernel = 'linear',random_state=0)
  svc_lin.fit(X_train, Y_train)

  #Using SVC method of svm class to use Kernel SVM Algorithm
  from sklearn.svm import SVC
  svc_rbf = SVC(kernel = 'rbf',random_state=0)
  svc_rbf.fit(X_train, Y_train)

  #Using GaussianNB method of naïve_bayes class to use Naïve Bayes Algorithm
  from sklearn.naive_bayes import GaussianNB
  gauss = GaussianNB()
  gauss.fit(X_train, Y_train)

  #Using DecisionTreeClassifier of tree class to use Decision Tree Algorithm
  from sklearn.tree import DecisionTreeClassifier
  tree = DecisionTreeClassifier(criterion = 'entropy',random_state=0)
  tree.fit(X_train, Y_train)

  #Using RandomForestClassifier method of ensemble class to use Random Forest Classification algorithm
  from sklearn.ensemble import RandomForestClassifier
  forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
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
    st.write("Congratulations! You made it.")
  else:
    st.write("Sorry! You didn't make it.")

st.markdown(body='''
   <div style='border-bottom:2px Solid Black'></div>
''', unsafe_allow_html=True)

#Training Models
st.header('Trainig Models')
data_load_state=st.text('Training in Progress...')
log,knn,svc_lin,svc_rbf,gauss,tree,forest=models(X_train, Y_train)
data_load_state.text('Models Trained Successfully!!')


#Show Performance of Models after Training
if(st.checkbox('Show Performance of Models')):
  st.subheader('Performance of Various Models on Same Dataset-')
  st.write('1.Logistic Regression Training Accuracy:', log.score(X_train, Y_train))
  confusionMatrix(log, "Logistic Regression")
  st.write('2. K Nearest Neighbor Training Accuracy:', knn.score(X_train, Y_train))
  confusionMatrix(knn,"K Nearest Neighbor Training Accuracy")
  st.write('3. Support Vector Machine (Linear Classifier) Training Accuracy:', svc_lin.score(X_train, Y_train))
  confusionMatrix(svc_lin,"Support Vector Machine (Linear Classifier)")
  st.write('4. Support Vector Machine (RBF Classifier) Training Accuracy:', svc_rbf.score(X_train, Y_train))
  confusionMatrix(svc_rbf,"Support Vector Machine (RBF Classifier)")
  st.write('5. Gaussian Naive Bayes Training Accuracy:', gauss.score(X_train, Y_train))
  confusionMatrix(gauss,"Gaussian Naive Bayes")
  st.write('6. Decision Tree Classifier Training Accuracy:', tree.score(X_train, Y_train))
  confusionMatrix(tree,"Decision Tree Classifier")
  st.write('7. Random Forest Classifier Training Accuracy:', forest.score(X_train, Y_train))
  confusionMatrix(forest,"Random Forest Classifier")
  st.markdown(body='''
  <h4><em>The best Model is Random Forest</em></h4>
  ''', unsafe_allow_html=True)

if(st.checkbox('Show Importance of Features in Predicting the Survival Status of Passengers')):
  importances = pd.DataFrame({'feature':titanic.iloc[:, 1:8].columns,'Importance':np.round(forest.feature_importances_,3)})
  importances = importances.sort_values('Importance',ascending=False).set_index('feature')
  importances

st.markdown(body='''
   <div style='border-bottom:2px Solid Black'></div>
''', unsafe_allow_html=True)

#Making Predictions 
st.write('\n')
st.header('Know your Survival Status (if you were on the Titanic)')
cl=st.text_input('Class')
sex=st.text_input('Sex')
age=st.text_input('Age')
sibsp=st.text_input('Number of Siblings/Spouses Aboard')
parch=st.text_input('Number of Parents/Children Aboard')
fare=st.text_input('Fare')

query=[[cl,sex,age,sibsp,parch,fare]]


if(st.button('Make Prediction')):
  make_prediction(query,forest)









