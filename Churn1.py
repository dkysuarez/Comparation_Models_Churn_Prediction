import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import plotly.express as px


st.set_page_config(page_title="Churn Model",
                   page_icon=":bar_chart:",
                   layout='wide'
                   )

# Remove menu
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
                """
st.markdown(hide_st_style, unsafe_allow_html=True)


# Detectando valores Missing Values,duplicados
missing_value = ["N/a", "na", np.nan]
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# Load the data set
df = pd.read_csv('Churn_Modelling.csv', na_values=missing_value)
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
df.dropna(inplace=True)


if st.sidebar.checkbox("Statics"):
    if st.sidebar.button("Head"):
        st.write(df.head(60))
    if st.sidebar.button("Tail"):
        st.write(df.tail(60))
    if st.sidebar.button("Columns"):
        st.write(df.columns)
    if st.sidebar.button("Describe"):
        st.write(df.describe(include='all'))
    if st.sidebar.button("Shape"):
        st.write("Number of Rows", df.shape[0])
        st.write("Number of Columns", df.shape[1])


col1, col2 = st.columns(2)
num_retained = df[df.Exited == 0.0].shape[0]
num_churned = df[df.Exited == 1.0].shape[0]
retined = num_retained / (num_retained + num_churned)*100
churned = num_churned / (num_retained + num_churned) * 100
col1.metric("customers stayed with the company:", retined, "%")
col2.metric("customers left with the company:", churned, "%", delta_color='inverse')

# plot churned
churnedCharts = st.sidebar.checkbox('Churned Charts')
if churnedCharts:
    opt = st.sidebar.radio('Churned:', [
        'Bar', 'Pie'
    ])
    if opt == 'Bar':
        plotchurned = df['Exited'].value_counts()
        st.bar_chart(plotchurned)
    if opt == 'Pie':

        # pie churn
        piechar = px.pie(df, names="Exited", title="Pie Churn")
        st.plotly_chart(piechar)


load = st.sidebar.checkbox('Load Charts')
if load:
    opt = st.sidebar.radio('Churned by type:', ["Gender", "Country", "Age", "Ternure"])
    if opt == 'Gender':
        # plot churned clients by gender
        fig = plt.figure(figsize=(9, 7))
        sns.countplot(x='Gender', hue='Exited', data=df)
        st.pyplot(plt)

    if opt == 'Country':
        # plot churned clients by country
        plt.figure(figsize=(15, 8))
        sns.countplot(x='Geography', hue='Exited', data=df)
        st.pyplot(plt)

    if opt == 'Age':
        # plot churned clients by age
        plt.figure(figsize=(18, 10))
        sns.countplot(x='Age', hue='Exited', data=df)
        st.pyplot(plt)

    if opt == 'Ternure':
        # plot churned clients by tenure
        plt.figure(figsize=(15, 10))
        sns.countplot(x='Tenure', hue='Exited', data=df)
        st.pyplot(plt)

df['Geography'].unique()
df = pd.get_dummies(df, drop_first=True)
X = df.drop('Exited', axis=1)
y = df['Exited']

X_res, y_res = SMOTE().fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.20, random_state=42)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Logistic Regression
log = LogisticRegression()
log.fit(X_train, y_train)
y_pred1 = log.predict(X_test)
accuracy_score(y_test, y_pred1)
precision_score(y_test, y_pred1)

# SVM
svm = svm.SVC()
svm.fit(X_train, y_train)
y_pred2 = svm.predict(X_test)

# KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred3 = knn.predict(X_test)


# DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred4 = dt.predict(X_test)


# RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred5 = rf.predict(X_test)


# GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
y_pred6 = gbc.predict(X_test)


loadModels = st.checkbox('Load Models')
if loadModels:
    opt = st.radio('Models:', ["Logistic Regression", "SVM", "KNeighborsClassifier",
                   "RandomForestClassifier", "GradientBoostingClassifier", "Comparation Models"])
    if opt == 'Logistic Regression':
        # Logistic Regression
        st.write("Logistic Regression")
        st.write("accuracy_score", accuracy_score(y_test, y_pred1))
        st.write("precision_score", precision_score(y_test, y_pred1))

    if opt == 'SVM':
        # SVM
        st.write("SVM")
        st.write("accuracy_score", accuracy_score(y_test, y_pred2))
        st.write("precision_score", precision_score(y_test, y_pred2))

    if opt == 'KNeighborsClassifier':
        # KNeighborsClassifier
        st.write("KNeighborsClassifier")
        st.write("accuracy_score", accuracy_score(y_test, y_pred3))
        st.write("precision_score", precision_score(y_test, y_pred3))

    if opt == 'DecisionTreeClassifier':
        # DecisionTreeClassifier
        st.write("DecisionTreeClassifier")
        st.write("accuracy_score", accuracy_score(y_test, y_pred4))
        st.write("precision_score", precision_score(y_test, y_pred4))

    if opt == 'RandomForestClassifier':
        # RandomForestClassifier
        st.write("RandomForestClassifier")
        st.write("precision_score", accuracy_score(y_test, y_pred5))
        st.write("accuracy_score", precision_score(y_test, y_pred5))

    if opt == 'GradientBoostingClassifier':
        # GradientBoostingClassifier
        st.write("GradientBoostingClassifier")
        st.write("precision_score", accuracy_score(y_test, y_pred6))
        st.write("accuracy_score", precision_score(y_test, y_pred6))
    if opt == 'Comparation Models':
        st.write("Comparation Models")
        final_data = pd.DataFrame({'Models': ['LR', 'SVC', 'KNN', 'DT', 'RF', 'GBC'],
                                   'ACC': [accuracy_score(y_test, y_pred1),
                                           accuracy_score(y_test, y_pred2),
                                           accuracy_score(y_test, y_pred3),
                                           accuracy_score(y_test, y_pred4),
                                           accuracy_score(y_test, y_pred5),
                                           accuracy_score(y_test, y_pred6)]})
        plt.figure(figsize=(15, 10))
        sns.barplot(x=final_data['Models'], y=final_data['ACC'], alpha=0.8)
        st.pyplot(plt)