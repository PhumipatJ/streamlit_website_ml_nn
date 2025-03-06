import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
import os
import seaborn as sns

st.markdown(
    """
    <style>
        .reportview-container {
            max-width: 100% !important;
        }

        .block-container {
            padding-left: 5rem;
            padding-right: 5rem;
            max-width: 90% !important;
        }

        .h1 {
            text-align: center;
            font-size: 48px;
            font-weight: bold;
        }

        .h2 {
            font-size: 36px;
        }

        .h3 {
            font-size: 24px;
        }

    </style>
    """,
    unsafe_allow_html=True
)

page = st.sidebar.selectbox("Select a Page", ["Machine Learning Detail", "Machine Learning Demo", "Neural Network Detail", "Neural Network Demo"])

if page == "Machine Learning Detail":
    def load_data(file):
        df = pd.read_csv(file)
        return df

    def load_data2(file):
        df = pd.read_csv(file)
        df.drop(columns=['ID', 'Title', 'Blurb', 'Crit', 'Crit_per_Level'], inplace=True)
        df['Partype'] = df['Partype'].apply(lambda x: x if x in ['Mana', 'Energy'] else 'None')
        df['Partype'] = df['Partype'].map({'Mana':2,'Energy':1,'None':0})
        df['Tags'] = df['Tags'].map({'Fighter':5,'Mage':4,'Assassin':3,'Marksman':2,'Tank':1,'Support':0})
        return df

    def print_model_metrics(model, X_train, y_train, model_name="Model"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_train)
        
        accuracy = accuracy_score(y_train, y_pred)
        precision = precision_score(y_train, y_pred, average='weighted')
        recall = recall_score(y_train, y_pred, average='weighted')
        f1 = f1_score(y_train, y_pred, average='weighted')
        conf_matrix = confusion_matrix(y_train, y_pred)
        
        # Display confusion matrix as a heatmap
        st.write(f"{model_name} - Confusion Matrix:")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', cbar=False, ax=ax, annot_kws={"size": 12})
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        st.pyplot(fig)
        
        st.write(f"{model_name} - Accuracy: {accuracy:.2f}")
        st.write(f"{model_name} - Precision: {precision:.2f}")
        st.write(f"{model_name} - Recall: {recall:.2f}")
        st.write(f"{model_name} - F1-Score: {f1:.2f}")

    st.markdown('<div class="h1">League of Legends Champions Tags Classifier</div>', unsafe_allow_html=True)
    uploaded_file = './champions.csv'
    df = load_data(uploaded_file)
    
    st.markdown('<br>', unsafe_allow_html=True)
    st.header("Raw Dataset", divider="green")
    st.dataframe(df, height=300 ,column_config={'Blurb': {'width': 100}})

    st.header("Exploratory Data Analysis (EDA)", divider="green")
    st.subheader("Feature Selection")
    row1col1, row1col2 = st.columns([0.4, 0.6])
    with row1col1:
        st.code("df.nunique() 'the number of unique values' ")
        st.dataframe(df.nunique(), width=550, height=275,column_config={'_index': {'width': 240}})

    ms = df.shape
    with row1col2:
        st.code(f"df.shape : {ms} -> {ms[0]} row {ms[1]} column 'the number of rows and columns in df'")
        st.write("&emsp;&emsp;`ID` , `Name` , `Title` and `Blurb` contain descriptive text or metadata which does not provide meaningful information for model training and is considered redundant for the analysis since it doesn't add predictive value.")
        st.write("&emsp;&emsp;`Crit` and `Crit_per_Level` are likely constant or have a limited set of values, it may not offer enough variation or significant predictive power compared to other variables like `Attack` , `Defense` , or `HP`.")
        st.code(f"df.drop(columns=['ID', 'Title', 'Name', 'Blurb', 'Crit', 'Crit_per_Level'])")
        st.write("&emsp;&emsp;By dropping these columns, we reduce noise and ensure that the data contains only relevant and meaningful features for further analysis, which is crucial for building effective machine learning models.")

    st.markdown('<br>', unsafe_allow_html=True)

    df.drop(columns=['ID','Name', 'Title', 'Blurb', 'Crit', 'Crit_per_Level'], inplace=True)

    st.subheader("Data Transformation")
    row2col1, row2col2, row2col3 = st.columns([0.3, 0.2, 0.5])
    with row2col1:
        st.code(" df.isnull().any() 'checks null values' ")
        st.dataframe(df.isnull().any(), width=550, height=275,column_config={'_index': {'width': 200}})

    ms = df.shape
    with row2col2:
        st.code(f" df['Partype'].unique() ")
        st.dataframe(df['Partype'].unique(), width=550, height=275)
    
    with row2col3:
        st.code(" df['Partype'] = df['Partype'].fillna('Other') ")
        st.write("&emsp;&emsp;Since `Partype` is categorical features, `None` can be replaced with a placeholder (e.g., `Unknown` or `Other`) to ensure no loss of information.")
        st.markdown('<br>', unsafe_allow_html=True)
        st.code(" df['Partype'] = df['Partype'].apply(lambda x: x if x in ['Mana', 'Energy'] else 'Special') ")
        st.write("&emsp;&emsp;In League of Legends (LoL), `Partype` refers to the type of resource a champion uses to cast abilities. There are three main types Mana , Energy and No Cost. Champions with No Cost abilities has special `Partype` (Fury, Rage, Heat, etc.) , so they can be classified under a Special category.")
    
    df['Partype'] = df['Partype'].fillna('Other')
    df['Partype'] = df['Partype'].apply(lambda x: x if x in ['Mana', 'Energy'] else 'Special')

    st.markdown('<br>', unsafe_allow_html=True)

    st.subheader("Data Formatting")
    row3col1, row3col2 = st.columns([0.5, 0.5])
    with row3col1:
        st.code(" df['Partype'] = df['Partype'].map({'Mana':2,'Energy':1,'Special':0}) ")
        col1, col2, col3 = st.columns([0.2, 0.1, 0.2])
    
        with col1:
            st.dataframe(df[['Partype']] , width=300, height=200)

        df['Partype'] = df['Partype'].map({'Mana':2,'Energy':1,'Special':0})

        with col2:
            st.markdown('<br><br>', unsafe_allow_html=True)
            st.header("&emsp;->")

        with col3:
            st.dataframe(df[['Partype']] , width=300, height=200)

    with row3col2:
        st.code(" df['Tags'] = df['Tags'].map({'Fighter':5,'Mage':4,'Assassin':3,'Marksman':2,'Tank':1,'Support':0}) ")
        col4, col5, col6 = st.columns([0.2, 0.1, 0.2])
        
        with col4:
            st.dataframe(df['Tags'], width=300, height=200)

        with col5:
            st.markdown('<br><br>', unsafe_allow_html=True)
            st.header("&emsp;->")

        df['Tags'] = df['Tags'].map({'Fighter':5,'Mage':4,'Assassin':3,'Marksman':2,'Tank':1,'Support':0})

        with col6:
            st.dataframe(df['Tags'], width=300, height=200)
    
    st.markdown('<br>', unsafe_allow_html=True)

    st.subheader("Data Scaling with MinMaxScaler()")
    row4col1, row4col2 = st.columns([0.6, 0.4])
    with row4col1:
        st.code("# Select MinMaxScaler() for scaling\nscaler = MinMaxScaler()\n# Exclude the 'Tags' column which is used as a label\ndf_scaled = df.drop(['Tags'], axis=1)\n# Apply MinMaxScaler()\ndf_scaled = pd.DataFrame(scaler.fit_transform(df_scaled), columns=df_scaled.columns)\n# Reattach the 'Tags' column\ndf_scaled['Tags'] = df['Tags'] ")

    with row4col2:
        st.write("&emsp;&emsp;In this process, `MinMaxScaler()` is used to scale the numerical features of the dataframe. The `MinMaxScaler()` scales the data to a range between 0 and 1.")
        st.write("&emsp;&emsp;The `Tags` column is label column that represents classes or categories. Scaling categorical variables would distort their meaning. So, we drop the `Tags` column before scaling, then reattach it later to ensure that the labels are preserved while the other features are normalized.")
    
    scaler = MinMaxScaler()
    df_scaled = df.drop(['Tags'], axis=1)
    df_scaled = pd.DataFrame(scaler.fit_transform(df_scaled), columns=df_scaled.columns)
    df_scaled['Tags'] = df['Tags']

    st.markdown('<br>', unsafe_allow_html=True)
    st.header("Dataset after EDA", divider="green")
    st.dataframe(df_scaled, height=300 )
    
    #X = df.drop(columns=['Tags','Name'])
    #y = df['Tags']
    #scaler = MinMaxScaler()
    #X_scaled = scaler.fit_transform(X)
    #X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
    #knn = KNeighborsClassifier(n_neighbors=5)
    #svm = SVC(probability=True)
    #ensemble_model = VotingClassifier(estimators=[('knn', knn), ('svm', svm)], voting='soft')

    #print(f"Training Data Metrics Ensemble:\n")
    #print_model_metrics(ensemble_model, X_train, y_train, model_name="Ensemble")
        
