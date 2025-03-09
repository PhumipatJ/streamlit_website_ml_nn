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
import random

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

    def model_evaluate(model, X_train, y_train, model_name="Model"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_train)
        
        accuracy = accuracy_score(y_train, y_pred)
        precision = precision_score(y_train, y_pred, average='weighted')
        recall = recall_score(y_train, y_pred, average='weighted')
        f1 = f1_score(y_train, y_pred, average='weighted')
        conf_matrix = confusion_matrix(y_train, y_pred)
        
        return conf_matrix, accuracy, precision, recall, f1

    st.markdown('<div class="h1">Machine Learning Model Development</div>', unsafe_allow_html=True)
    uploaded_file = './champions.csv'
    df = load_data(uploaded_file)

    st.markdown('<br>', unsafe_allow_html=True)
    st.header("Machine Learning Model for LoL Champion Role Classification", divider="green")
    st.subheader("What is League of Legends ?")
    st.write("&emsp;&emsp;League of Legends (LoL) is a Multiplayer Online Battle Arena (MOBA) game developed by Riot Games, similar to other well-known titles like DOTA 2 and Arena of Valor (RoV) , where players control characters known as `champions` and work together in teams to compete against other players. The game features a variety of champions, each with unique abilities and playstyles, and players are tasked with defeating the opposing team by destroying their base while protecting their own. The champions are classified into different roles based on their abilities and responsibilities during the game. These roles include.")
    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        st.markdown("""
        - `Fighter` : Durable and capable of dealing high damage in close combat.
        - `Assassin` : High burst damage dealers focused on eliminating key targets.
        - `Mage` : Champions that use powerful spells to deal area-of-effect or single-target damage.
        - `Marksman` : Ranged attackers that deal consistent damage from a distance.
        - `Tank` : Champions with high health and defensive abilities, focused on absorbing damage.
        - `Support` : Champions that assist their team through healing, crowd control, or utility abilities.
        """)

    with col2:
        image_folder = 'assets/LoL'
        sample_image_path = os.path.join(image_folder, 'logo', f"LoL.webp")
        st.image(sample_image_path)

    st.markdown('<br>', unsafe_allow_html=True)
    st.subheader("How does Machine Learning help in classifying LoL Champions Role ?")
    st.write("&emsp;&emsp;Machine Learning (ML) is a field of artificial intelligence that allows systems to learn patterns from data without being explicitly programmed. In the context of League of Legends, machine learning models can be employed to automate the classification of champions into their respective roles (e.g., `Fighter`, `Assassin`, `Mage`, etc.), based on various features of the champion, such as their stats, abilities, and in-game behavior.")

    st.markdown('<br>', unsafe_allow_html=True)
    st.header("Data Collection", divider="green")
    st.write("&emsp;&emsp;The data used to train and classify League of Legends champions into their respective roles is collected from [Riot Games Developer Portal](https://developer.riotgames.com/docs/lol). Specifically, the data comes from an [API endpoint](https://ddragon.leagueoflegends.com/cdn/15.5.1/data/en_US/champion.json) provided by Riot Games, which allows developers to access detailed information about the game, including champion statistics, abilities, and roles. The API data can be retrieved in a structured JSON format, I reformatted the data retrieved from the API into a CSV format, making it easier to process and feed into the model for training.")
    st.subheader("Raw Dataset")
    st.dataframe(df, height=300 ,column_config={'Blurb': {'width': 100}})
    features = [
    {"ID": "A unique identifier for each champion in the dataset."},
    {"Name": "The name of the champion, such as 'Ahri' , 'Garen' or 'Yasuo.'"},
    {"Title": "A brief descriptor of the champion’s persona or role in the game."},
    {"Blurb": "A short narrative or background story about the champion."},
    {"Partype": "The champion's resource system (mana, energy, etc.)."},
    {"Attack": "The champion’s ability to deal damage with basic attacks."},
    {"Defense": "The champion’s ability to take damage and survive in battles."},
    {"Magic": "A measure of how effective the champion is at dealing magic (spell) damage."},
    {"Difficulty": "How challenging a champion is to play."},
    {"HP": "The initial champion’s total health."},
    {"HP_per_Level": "The amount of health the champion gains as they level up."},
    {"MP": "The initial amount of mana for champion to cast their abilities."},
    {"MP_per_Level": "The amount of mana the champion gains as they level up."},
    {"Move_Speed": "The initial champion's movement speed."},
    {"Armor": "The initial champion’s ability to resist physical damage."},
    {"Armor_per_Level": "The amount of armor the champion gains as they level up."},
    {"Spell_Block": "The initial champion’s ability to resist magic damage or crowd control effects."},
    {"Spell_Block_per_Level": "The amount of spell block the champion gains as they level up."},
    {"Attack_Range": "How far the champion can attack."},
    {"HP_Regen": "The initial rate at which the champion's health regenerates over time."},
    {"HP_Regen_per_Level": "The rate at which the champion’s health regeneration improves as they level up."},
    {"MP_Regen": "The initial rate at which the champion’s mana regenerates over time."},
    {"MP_Regen_per_Level": "The rate at which the champion’s mana regeneration improves as they level up."},
    {"Crit": "The initial champion’s critical hit chance."},
    {"Crit_per_Level": "The increase in critical hit chance as the champion levels up."},
    {"Attack_Damage": "The initial base damage dealt by the champion with basic attacks."},
    {"Attack_Damage_per_Level": "The increase in attack damage the champion gets as they level up."},
    {"Attack_Speed_per_Level": "How much faster the champion's attacks become as they level up."},
    {"Attack_Speed": "Initial value represent how fast champion can perform basic attacks."},
    {"Tags": "Represents the champion’s general role or archetype in the game. This will be use as a label"}
    ]
    st.subheader("Feature Explanation")
    st.write(features)

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

    df.drop(columns=['ID', 'Title', 'Blurb', 'Crit', 'Crit_per_Level'], inplace=True)

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

    st.subheader("Data Scaling")
    row4col1, row4col2 = st.columns([0.6, 0.4])
    with row4col1:
        st.code("# Select MinMaxScaler() for scaling\nscaler = MinMaxScaler()\n# Exclude the 'Tags' column which is used as a label\ndf_scaled = df.drop(['Tags'], axis=1)\n# Apply MinMaxScaler()\ndf_scaled = pd.DataFrame(scaler.fit_transform(df_scaled), columns=df_scaled.columns)\n# Reattach the 'Tags' column\ndf_scaled['Tags'] = df['Tags'] ")

    with row4col2:
        st.write("&emsp;&emsp;In this process, `MinMaxScaler()` is used to scale the numerical features of the dataframe. The `MinMaxScaler()` scales the data to a range between 0 and 1.")
        st.write("&emsp;&emsp;The `Tags` column is label column that represents classes or categories. Scaling categorical variables would distort their meaning. So, we drop the `Tags` column before scaling, then reattach it later to ensure that the labels are preserved while the other features are normalized.")
    
    scaler = MinMaxScaler()
    df_scaled = df.drop(['Tags','Name'], axis=1)
    df_scaled = pd.DataFrame(scaler.fit_transform(df_scaled), columns=df_scaled.columns)
    df_scaled['Tags'] = df['Tags']
    df_scaled['Name'] = df['Name']

    st.markdown('<br>', unsafe_allow_html=True)
    st.header("Dataset after EDA", divider="green")
    st.dataframe(df_scaled.drop(columns=['Name']), height=300)
    
    st.markdown('<br>', unsafe_allow_html=True)
    st.header("Model Option for Classification", divider="green")
    col1, col2 = st.columns([0.5, 0.5])
    with col1:
        st.subheader("K-Nearest Neighbors (KNN)")
        st.write("&emsp;&emsp;KNN is a simple, non-parametric method used for classification tasks. It classifies data points based on the majority class of their nearest neighbors in the feature space. For this classification problem, KNN would classify champions into roles (e.g., Fighter, Assassin, Mage) by comparing the champion’s features to those of others.")

    with col2:
        st.subheader("Hyperparameters")
        st.markdown("""
        - `K (number of neighbors)` : Determines how many nearest neighbors are considered when making a classification decision.
        - `Distance Metric` : The measure used to calculate the distance between data points (e.g., Euclidean, Manhattan).
        """)
    
    st.markdown('<br>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([0.5, 0.5])
    with col1:
        st.subheader("Support Vector Machine (SVM)")
        st.write("&emsp;&emsp;SVM is a powerful algorithm for classification tasks, especially when the data is not linearly separable. It works by finding a hyperplane that best divides data points of different classes. In the context of champion role classification, SVM would aim to find an optimal decision boundary to separate different roles.")

    with col2:
        st.subheader("Hyperparameters")
        st.markdown("""
        - `Kernel` : Defines the function used to map input data into higher dimensions. Common choices include linear, polynomial, and RBF kernels.
        - `C` : Regularization parameter that controls the trade-off between achieving a low error on the training data and having a simple decision boundary.
        """)
    
        st.markdown('<br>', unsafe_allow_html=True)

    col1, col2 = st.columns([0.5, 0.5])
    with col1:
        st.subheader("Ensemble Methods")
        st.write("&emsp;&emsp;Ensemble methods combine multiple models to improve classification performance. A voting classifier, for instance, combines predictions from multiple base models (e.g., KNN, SVM, Decision Trees) and classifies champions based on the majority vote or probability average from these models.")

    with col2:
        st.subheader("Hyperparameters")
        st.markdown("""
        - `Base Models` : The type of models used within the ensemble (e.g., KNN, SVM).
        - `Voting Strategy` : Determines how the predictions from base models are combined. It can be "hard" (majority vote) or "soft" (probability average).
        """)



    st.markdown('<br>', unsafe_allow_html=True)
    st.header("Model Preparation", divider="green")
    

    row5col1, row5col2 = st.columns([0.5, 0.5])
    with row5col1:
        st.subheader("1. Feature and Target Selection")
        st.code("X = df_scaled.drop(['Name', 'Tags'], axis=1)  # Features (X)\ny = df_scaled['Tags']  # Target labels (y)")
        st.markdown("- Selecting features (X) and target (y) for model training.")
        st.markdown('<br>', unsafe_allow_html=True)
        st.subheader("3. Model Initialization")
        st.code("knn = KNeighborsClassifier(n_neighbors=5)\nsvm = SVC(kernel='linear', C=1, probability=True, random_state=42)")
        st.markdown("- Initializing models KNeighborsClassifier (KNN) and SVC (Support Vector Machine)")

    with row5col2:
        st.subheader("2. Data Splitting")
        st.code("X_train, X_test, y_train, y_test = \ntrain_test_split(X, y, test_size=0.2, random_state=42)")
        st.markdown("- Randomly splitting the dataset into training (80%) and testing (20%) sets.")
        st.markdown('<br>', unsafe_allow_html=True)
        st.subheader("4. Ensemble Model Initialization")
        st.code("ensemble_model = \nVotingClassifier(estimators=[('knn', knn), ('svm', svm)], voting='hard')")
        st.markdown("- Creating an ensemble model, which combines KNN and SVM for better predictions.")

    st.markdown('<br>', unsafe_allow_html=True)
    st.header("Model Evaluation", divider="green")
    st.code("""
    def model_evaluate(model, X_data, y_data, model_name="Model"):
        model.fit(X_data, y_data)  # Train the model using the training data
        y_pred = model.predict(X_data)  # Predict the labels for the training data

        accuracy = accuracy_score(y_data, y_pred)  # Calculate the accuracy (overall correct predictions)
        precision = precision_score(y_data, y_pred, average='weighted')  # Calculate precision (correct positives out of predicted positives)
        recall = recall_score(y_data, y_pred, average='weighted')  # Calculate recall (correct positives out of actual positives)
        f1 = f1_score(y_data, y_pred, average='weighted')  # Calculate F1 score (harmonic mean of precision and recall)
        conf_matrix = confusion_matrix(y_data, y_pred)  # Calculate confusion matrix to show true vs predicted labels
    """)

    X = df_scaled.drop(['Name', 'Tags'], axis=1) 
    y = df_scaled['Tags']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5)
    svm = SVC(kernel='linear', C=1,probability=True, random_state=42)  
    ensemble_model = VotingClassifier(estimators=[('knn', knn), ('svm', svm)], voting='hard')

    st.markdown('<br>', unsafe_allow_html=True)
    row6col1, row6col2 = st.columns([0.5, 0.5])
    with row6col1:
        st.subheader("Training Data Metrics - KNN")
        conf_matrix, accuracy, precision, recall, f1 = model_evaluate(knn, X_train, y_train, model_name="KNN")
        st.code(f"""model_evaluate(knn, X_train, y_train, model_name="KNN")\n\nConfusion Matrix:\n{conf_matrix}\nAccuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1-Score: {f1}""")
        
        st.markdown('<br>', unsafe_allow_html=True)

        st.subheader("Training Data Metrics - SVM")
        conf_matrix, accuracy, precision, recall, f1 = model_evaluate(svm, X_train, y_train, model_name="SVM")
        st.code(f"""model_evaluate(svm, X_train, y_train, model_name="SVM")\n\nConfusion Matrix:\n{conf_matrix}\nAccuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1-Score: {f1}""")

        st.markdown('<br>', unsafe_allow_html=True)

        st.subheader("Training Data Metrics - Ensemble (KNN + SVM)")
        conf_matrix, accuracy, precision, recall, f1 = model_evaluate(ensemble_model, X_train, y_train, model_name="Ensemble")
        st.code(f"""model_evaluate(ensemble_model, X_train, y_train, model_name="Ensemble")\n\nConfusion Matrix:\n{conf_matrix}\nAccuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1-Score: {f1}""")

    with row6col2:
        st.subheader("Testing Data Metrics - KNN")
        conf_matrix, accuracy, precision, recall, f1 = model_evaluate(knn, X_test, y_test, model_name="KNN")
        st.code(f"""model_evaluate(knn, X_test, y_test, model_name="KNN")\n\nConfusion Matrix:\n{conf_matrix}\nAccuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1-Score: {f1}""")

        st.markdown('<br>', unsafe_allow_html=True)

        st.subheader("Testing Data Metrics - SVM")
        conf_matrix, accuracy, precision, recall, f1 = model_evaluate(svm, X_test, y_test, model_name="SVM")
        st.code(f"""model_evaluate(svm, X_test, y_test, model_name="SVM")\n\nConfusion Matrix:\n{conf_matrix}\nAccuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1-Score: {f1}""")

        st.markdown('<br>', unsafe_allow_html=True)

        st.subheader("Testing Data Metrics - Ensemble (KNN + SVM)")
        conf_matrix, accuracy, precision, recall, f1 = model_evaluate(ensemble_model, X_test, y_test, model_name="Ensemble")
        st.code(f"""model_evaluate(ensemble_model, X_test, y_test, model_name="Ensemble")\n\nConfusion Matrix:\n{conf_matrix}\nAccuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1-Score: {f1}""")

elif page == "Machine Learning Demo":
    def load_data(file):
        df = pd.read_csv(file)
        return df

    uploaded_file = './champions.csv'
    df = load_data(uploaded_file)
    df.drop(columns=['ID', 'Title', 'Blurb', 'Crit', 'Crit_per_Level'], inplace=True)

    df['Partype'] = df['Partype'].fillna('Other')
    df['Partype'] = df['Partype'].apply(lambda x: x if x in ['Mana', 'Energy'] else 'Special')
    df['Partype'] = df['Partype'].map({'Mana':2,'Energy':1,'Special':0})
    df['Tags'] = df['Tags'].map({'Fighter':5,'Mage':4,'Assassin':3,'Marksman':2,'Tank':1,'Support':0})
    scaler = MinMaxScaler()
    df_scaled = df.drop(['Tags','Name'], axis=1)
    df_scaled = pd.DataFrame(scaler.fit_transform(df_scaled), columns=df_scaled.columns)
    df_scaled['Tags'] = df['Tags']
    df_scaled['Name'] = df['Name']
    X = df_scaled.drop(['Name', 'Tags'], axis=1) 
    y = df_scaled['Tags']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5)
    svm = SVC(kernel='linear', C=1,probability=True, random_state=42)  
    ensemble_model = VotingClassifier(estimators=[('knn', knn), ('svm', svm)], voting='hard')

    knn.fit(X_test, y_test)

    svm.fit(X_test, y_test)

    ensemble_model.fit(X_test, y_test)

    st.markdown('<div class="h1">League of Legends Champions Role Classifier</div>', unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html=True)
    
    st.header("Guess the Role of a Random Champion !", divider="green")
    row7col1, row7col2 = st.columns([0.3, 0.7])
    sample_image_path = ""
    true_label_image_path = ""
    predicted_label_image_path = ""

    row7col1, row7col2 = st.columns([0.3, 0.7])
    sample_image_path = ""
    true_label_image_path = ""
    predicted_label_image_path = ""

    with row7col1:
        with st.form(key='model_selection_form'):
            model_option = st.selectbox('Choose a model', ['KNN', 'SVM', 'Ensemble'])
            dataset_option = st.selectbox('Choose a dataset', ['Train', 'Test'])
            submit_button = st.form_submit_button(label='Random Champions')

    if submit_button:
        if dataset_option == 'Train':
            X_data = X_train
            y_data = y_train
        else:
            X_data = X_test
            y_data = y_test

        random_index = random.choice(X_data.index)
        sample_features = X_data.loc[random_index]
        name = df.loc[random_index, 'Name']
        true_label = y_data.loc[random_index]

        if model_option == 'KNN':
            predicted_label = knn.predict(sample_features.to_frame().T)
        elif model_option == 'SVM':
            predicted_label = svm.predict(sample_features.to_frame().T)
        else:
            predicted_label = ensemble_model.predict(sample_features.to_frame().T)
            
        category_map = ['Support', 'Tank', 'Marksman', 'Assasin', 'Mage', 'Fighter']
        image_folder = 'assets/LoL'
        sample_image_path = os.path.join(image_folder, 'champion', f"{name}_0.jpg")
        true_label_image_path = os.path.join(image_folder, 'role', f"{category_map[true_label]}.png")
        predicted_label_image_path = os.path.join(image_folder, 'role', f"{category_map[predicted_label[0]]}.png")

    with row7col2:
        try:
            st.dataframe(sample_features, height=250,column_config={'_index': {'width': 300}})
        except Exception as e:
            print(e)
    
    st.markdown('<br>', unsafe_allow_html=True)

    col1, col2, col3, col4, col5, col6, col7 = st.columns([0.05, 0.3, 0.05, 0.05 ,0.25, 0.05, 0.25])
    try:
        with col2:
            st.image(sample_image_path, caption=f"Sample : {name}", width=300)

        with col5:
            st.write(" ")
            st.write(" ")
            st.image(true_label_image_path, caption=f"True Label : {category_map[true_label]}", width=192)

        with col7:
            st.write(" ")
            st.write(" ")
            st.image(predicted_label_image_path, caption=f"Predicted Label : {category_map[predicted_label[0]]}", width=192)

    except Exception as e:
        print(e)
    
    st.markdown('<br>', unsafe_allow_html=True)
    st.header("Game Just Released a New Champion! Guess Their Role!", divider="green")

    def getMin(column_name):
        return df[column_name].min()

    def getMax(column_name):
        return df[column_name].max()
    
    
    st.subheader("Select Input")
    col1, col2= st.columns([0.5,0.5])
    with col1:
        attack = st.slider("Attack", min_value=getMin('Attack'), max_value=getMax('Attack'))
        defense = st.slider("Defense", min_value=getMin('Defense'), max_value=getMax('Defense'))
        hp = st.slider("HP", min_value=getMin('HP'), max_value=getMax('HP'))
        mp = st.slider("MP", min_value=getMin('MP'), max_value=530)
        armor = st.slider("Armor", min_value=getMin('Armor'), max_value=getMax('Armor'))
        spell_block = st.slider("Spell Block", min_value=getMin('Spell_Block'), max_value=getMax('Spell_Block'))
        hp_regen = st.slider("HP Regen", min_value=getMin('HP_Regen'), max_value=getMax('HP_Regen'))
        mp_regen = st.slider("MP Regen", min_value=getMin('MP_Regen'), max_value=getMax('MP_Regen'))
        attack_damage = st.slider("Attack Damage", min_value=getMin('Attack_Damage'), max_value=getMax('Attack_Damage'))
        attack_speed = st.slider("Attack Speed", min_value=getMin('Attack_Speed'), max_value=getMax('Attack_Speed'))
        attack_range = st.slider("Attack Range", min_value=getMin('Attack_Range'), max_value=getMax('Attack_Range'))
        partype = st.selectbox("Select Partype", ["Mana", "Energy", "Special"])
    
    with col2:
        magic = st.slider("Magic", min_value=getMin('Magic'), max_value=getMax('Magic'))
        difficulty = st.slider("Difficulty", min_value=getMin('Difficulty'), max_value=getMax('Difficulty'))
        hp_per_level = st.slider("HP per Level", min_value=getMin('HP_per_Level'), max_value=getMax('HP_per_Level'))
        mp_per_level = st.slider("MP per Level", min_value=getMin('MP_per_Level'), max_value=getMax('MP_per_Level'))
        armor_per_level = st.slider("Armor per Level", min_value=getMin('Armor_per_Level'), max_value=getMax('Armor_per_Level'))
        spell_block_per_level = st.slider("Spell Block per Level", min_value=getMin('Spell_Block_per_Level'), max_value=getMax('Spell_Block_per_Level'))
        hp_regen_per_level = st.slider("HP Regen per Level", min_value=getMin('HP_Regen_per_Level'), max_value=getMax('HP_Regen_per_Level'))
        mp_regen_per_level = st.slider("MP Regen per Level", min_value=getMin('MP_Regen_per_Level'), max_value=getMax('MP_Regen_per_Level'))
        attack_damage_per_level = st.slider("Attack Damage per Level", min_value=getMin('Attack_Damage_per_Level'), max_value=getMax('Attack_Damage_per_Level'))
        attack_speed_per_level = st.slider("Attack Speed per Level", min_value=getMin('Attack_Speed_per_Level'), max_value=getMax('Attack_Speed_per_Level'))
        move_speed = st.slider("Move Speed", min_value=getMin('Move_Speed'), max_value=getMax('Move_Speed'))

    col1, col2, col3 = st.columns([0.25, 0.5, 0.25])
    with col2:
        with st.form(key='model_selection_form_guess'):
            model_option = st.selectbox('Choose a model', ['KNN', 'SVM', 'Ensemble'])
            submit_button = st.form_submit_button(label='Guess Champions Role')

            if submit_button:
                sample_features_guess = pd.DataFrame([{
                    "Partype": 1 if partype == "Mana" else (0.5 if partype == "Energy" else 0),
                    "Attack": attack,
                    "Defense": defense,
                    "Magic": magic,
                    "Difficulty": difficulty,
                    "HP": hp,
                    "HP_per_Level": hp_per_level,
                    "MP": mp,
                    "MP_per_Level": mp_per_level,
                    "Move_Speed": move_speed,
                    "Armor": armor,
                    "Armor_per_Level": armor_per_level,
                    "Spell_Block": spell_block,
                    "Spell_Block_per_Level": spell_block_per_level,
                    "Attack_Range": attack_range,
                    "HP_Regen": hp_regen,
                    "HP_Regen_per_Level": hp_regen_per_level,
                    "MP_Regen": mp_regen,
                    "MP_Regen_per_Level": mp_regen_per_level,
                    "Attack_Damage": attack_damage,
                    "Attack_Damage_per_Level": attack_damage_per_level,
                    "Attack_Speed_per_Level": attack_speed_per_level,
                    "Attack_Speed": attack_speed
                }])    

                if model_option == 'KNN':
                    predicted_label = knn.predict(sample_features_guess)
                elif model_option == 'SVM':
                    predicted_label = svm.predict(sample_features_guess)
                else:
                    predicted_label = ensemble_model.predict(sample_features_guess)

                category_map = ['Support', 'Tank', 'Marksman', 'Assassin', 'Mage', 'Fighter']
                try:
                    col1, col2, col3, col4 = st.columns([0.025, 0.2, 0.05, 0.2])
                    image_folder = 'assets/LoL'

                    with col2:
                        predicted_label_image_anon = os.path.join(image_folder, 'role', "anonymous.webp")
                        st.image(predicted_label_image_anon, caption=f"New Champion", width=192)

                    with col4:
                        predicted_label_image = os.path.join(image_folder, 'role', f"{category_map[int(predicted_label[0])]}.png")
                        st.image(predicted_label_image, caption=f"Might be : {category_map[int(predicted_label[0])]}", width=192)
                except Exception as e:
                    print(e)

elif page == "Neural Network Detail":
    st.markdown('<div class="h1">Neural Network Model Development</div>', unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html=True)
    st.header("Neural Network for Valorant Weapon Skin Recognition using CNN", divider="green")

    col1, col2 = st.columns([0.5, 0.5])

    with col1:
        st.subheader("What is Valorant ?")
        st.write("&emsp;&emsp;Valorant is a tactical first-person shooter (FPS) game developed by Riot Games. It features 5v5 matches where players select unique characters called Agents, each with their own abilities. The game emphasizes strategy, precise shooting, and teamwork.")
        st.subheader("What are Valorant Weapon Skins ?")
        st.write("&emsp;&emsp;In Valorant, players can customize their weapons with skins—cosmetic changes that alter a weapon’s look, sound, and animations while some exclusive skins may change their appearance. There are 19 weapon types, each with multiple skins, some featuring variants or upgrades. While skins don’t affect gameplay, identifying them can be difficult.")
        

    with col2:
        image_folder = 'assets/Valorant'
        sample_image_path = os.path.join(image_folder, 'valorant', "valorant.webp")
        st.image(sample_image_path)
    
    st.markdown('<br>', unsafe_allow_html=True)

    st.header("How does CNN help in classifying Valorant Weapon Skins ?", divider="green")
    st.write("&emsp;&emsp;A Convolutional Neural Network (CNN) processes images through multiple layers to extract and classify features. For Valorant weapon skins, a CNN can differentiate skins based on colors, patterns, and textures. Let’s break down how each layer contributes to classification.")
    
    col1, col2 = st.columns([0.5, 0.5])
    with col1:
        st.subheader("1. Convolutional Layer")
        st.write("""
        Role: Detects edges, shapes, textures, and patterns.
        - Uses filters (kernels) to scan the image and extract important details.
        - Early layers detect edges and outlines of weapons.
        - Deeper layers learn textures and patterns unique to each skin.
        """)

    with col2:
        st.subheader("2. Pooling Layer ")
        st.write("""
        Role: Reduces the spatial size of the image while keeping key features.
        - Max Pooling selects the most prominent features.
        - Average Pooling smooths the image.
        """)
    
    st.markdown('<br>', unsafe_allow_html=True)

    col1, col2 = st.columns([0.5, 0.5])
    with col1:
        st.subheader("3. Flatten Layer")
        st.write("""
        Role: Converts the 2D image matrix into a 1D feature vector.
        - Prepares extracted features for the next stage (fully connected layer).
        """)
    
    with col2:
        st.subheader("4. Fully Connected Layer")
        st.write("""
        Role: Uses the extracted features to classify the image.
        - Each neuron represents a specific skin category.
        - Uses Softmax activation to predict the probability of each skin.
        """)

    st.header("Data Collection", divider="green")
    st.write("&emsp;&emsp;The data used to classify Valorant weapon skins is collected from the [Riot Games Developer Portal](https://developer.riotgames.com/docs/valorant). Specifically, the skins and related assets come from the Public Content Catalog which can be downloaded for free, which provides various Valorant assets. Weapon skin images (stored in `.png` format) also included but there's one problem.")

    st.markdown('<br>', unsafe_allow_html=True)
    st.subheader("Processing the Weapon Skin Data")
    st.write("""
    &emsp;&emsp;When downloaded, all 19 weapon skins are stored in a single folder as `.png` files. However, the file names are random IDs.

    - `10B39A6F-494C-CA3C-497F-24877AB43A47.png`
    - `27C1C5E0-4829-DFB2-9085-33BC04A73971.png`
    - `AA1728CF-41B5-3C90-99A8-F9857CB083F4.png`

    &emsp;&emsp;These filenames do not provide any useful information about the weapon or skin name. Because of this, I have to manually open each image one by one and classify them into separate folders based on their weapon type.

    - `/phantom/` → (All Phantom skins go here)
    - `/vandal/` → (All Vandal skins go here)
    - `/spectre/` → (All Spectre skins go here)

    &emsp;&emsp;This manual classification process is necessary to correctly label the images before training the Convolutional Neural Network (CNN) for weapon skin recognition. After sorting, the images are further processed and augmented to improve model accuracy.
    """)

    st.markdown('<br>', unsafe_allow_html=True)
    st.subheader("The Real Struggle")
    st.write("&emsp;&emsp;Manually sorting hundreds of randomly named images was an absolute nightmare. It took me 3 hours to finish this process. If I had a CNN model already trained, it probably could have done this in seconds. But no—here I am, manually dragging and dropping files like it’s the Stone Age of AI. 😭")


    st.header("Valorant Weapons Skin Dataset", divider="green")
    col1, col2,col3, col4 = st.columns([0.25, 0.25, 0.25, 0.25])
    with col1:
        original_skin = st.button("Weapons without Skin")
    with col2:
        with_skin = st.button("Weapons with Skin")

    if original_skin:
        skin_type = "originalSkin"
    elif with_skin:
        skin_type = "withSkin"
    else:
        skin_type = "originalSkin" 

    st.markdown('<br>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns([0.26, 0.2, 0.25, 0.25])
    with col1:
        st.image(os.path.join(image_folder, skin_type, "ghost.png"),caption="Ghost",width=200)
    with col2:
        st.image(os.path.join(image_folder, skin_type, "sheriff.png"),caption="Sheriff",width=150)
    with col3:
        st.image(os.path.join(image_folder, skin_type, "ares.png"),caption="Ares",width=200)
    with col4: 
        st.image(os.path.join(image_folder, skin_type, "odin.png"),caption="Odin",width=200)

    st.markdown('<br><br>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns([0.25, 0.25, 0.25, 0.25])
    with col1:
        st.image(os.path.join(image_folder, skin_type, "shorty.png"),caption="Shorty",width=200)
    with col2:
        st.image(os.path.join(image_folder, skin_type, "judge.png"),caption="Judge",width=200) 
    with col3:
        st.image(os.path.join(image_folder, skin_type, "stinger.png"),caption="Stinger",width=200)
    with col4: 
        st.image(os.path.join(image_folder, skin_type, "spectre.png"),caption="Spectre",width=200)
    
    st.markdown('<br><br>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns([0.25, 0.25, 0.25, 0.25])
    with col1:
        st.image(os.path.join(image_folder, skin_type, "bucky.png"),caption="Bucky",width=200)
    with col2:
        st.image(os.path.join(image_folder, skin_type, "phantom.png"),caption="Phantom",width=200) 
    with col3:
        st.image(os.path.join(image_folder, skin_type, "vandal.png"),caption="Vandal",width=200)
    with col4: 
        st.image(os.path.join(image_folder, skin_type, "bulldog.png"),caption="Bulldog",width=200)
    
    st.markdown('<br><br>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns([0.25, 0.25, 0.25, 0.25])
    with col1:
        st.image(os.path.join(image_folder, skin_type, "marshal.png"),caption="Marshal",width=200)
    with col2:
        st.image(os.path.join(image_folder, skin_type, "guardian.png"),caption="Guardian",width=200) 
    with col3:
        st.image(os.path.join(image_folder, skin_type, "outlaw.png"),caption="Outlaw",width=200)
    with col4: 
        st.image(os.path.join(image_folder, skin_type, "operator.png"),caption="Operator",width=200)
    
    st.markdown('<br><br>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns([0.1, 0.2, 0.2, 0.25])
    with col2:
        st.image(os.path.join(image_folder, skin_type, "classic.png"),caption="Classic",width=150)
    with col3:
        st.image(os.path.join(image_folder, skin_type, "frenzy.png"),caption="Frenzy",width=150) 
    with col4:
        st.image(os.path.join(image_folder, skin_type, "melee.png"),caption="Melee",width=175)

    st.markdown('<br>', unsafe_allow_html=True)
    st.header("Exploratory Data Analysis (EDA)", divider="green")
    
    