# -*- coding: utf-8 -*-
"""ML_LoL_Champions.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1qtR5r7rZVSA3i15ZRCVekof5JMH7vJ8J
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report

from google.colab import drive
drive.mount('/content/drive')

csv_file_path = '/content/drive/MyDrive/LoL_champion_dataset/champions.csv'

df = pd.read_csv(csv_file_path)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

df.head()

df.nunique()

df.drop(columns=['ID', 'Title', 'Blurb', 'Crit', 'Crit_per_Level'], inplace=True)
df.head()

df['Partype'].unique()

df['Partype'] = df['Partype'].apply(lambda x: x if x in ['Mana', 'Energy'] else 'None')
df['Partype'].unique()

df['Partype'] = df['Partype'].map({'Mana':2,'Energy':1,'None':0})
df['Tags'] = df['Tags'].map({'Fighter':5,'Mage':4,'Assassin':3,'Marksman':2,'Tank':1,'Support':0})

# Exclude the 'Name' column and scale the remaining columns
scaler = MinMaxScaler()
# Exclude the 'Tags' column and scale the remaining columns
df_scaled = df.drop(['Name', 'Tags'], axis=1)
# Apply MinMaxScaler
df_scaled = pd.DataFrame(scaler.fit_transform(df_scaled), columns=df_scaled.columns)
# Reattach the 'Name' column
df_scaled['Name'] = df['Name']
df_scaled['Tags'] = df['Tags']

df_scaled.head()

# Assuming df contains the features (X) and target labels (y)
X = df_scaled.drop(['Name', 'Tags'], axis=1)  # Drop 'Name' and 'tags' column
y = df_scaled['Tags']  # Target labels

# Split the data into training and test sets (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize KNN and SVM models
knn = KNeighborsClassifier(n_neighbors=5)
svm = SVC(kernel='linear', C=1,probability=True, random_state=42)  # Use probability=True for soft voting

# Initialize the VotingClassifier with hard voting (default)
ensemble_model = VotingClassifier(estimators=[('knn', knn), ('svm', svm)], voting='hard')

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Function to print model metrics
def print_model_metrics(model, X_train, y_train, model_name="Model"):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)

    accuracy = accuracy_score(y_train, y_pred)
    precision = precision_score(y_train, y_pred, average='weighted')
    recall = recall_score(y_train, y_pred, average='weighted')
    f1 = f1_score(y_train, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_train, y_pred)

    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}\n")

print(f"Training Data Metrics KNN:\n")
print_model_metrics(knn, X_train, y_train, model_name="KNN")

print(f"Testing Data Metrics KNN:\n")
print_model_metrics(knn, X_test, y_test, model_name="KNN")

"""--------------------------------------------------------------------------------"""

print(f"Training Data Metrics SVM:\n")
print_model_metrics(svm, X_train, y_train, model_name="SVM")

print(f"Testing Data Metrics SVM:\n")
print_model_metrics(svm, X_test, y_test, model_name="SVM")

"""--------------------------------------------------------------------------------"""

print(f"Training Data Metrics Ensemble:\n")
print_model_metrics(ensemble_model, X_train, y_train, model_name="Ensemble")

print(f"Testing Data Metrics Ensemble:\n")
print_model_metrics(ensemble_model, X_test, y_test, model_name="Ensemble")

import pandas as pd

# Map for category labels
category_map = {
    'Fighter': 5,
    'Mage': 4,
    'Assassin': 3,
    'Marksman': 2,
    'Tank': 1,
    'Support': 0
}

# Prepare a list to store results
results = []

# Loop through the test set
for i in range(len(X_test)):
    # Get the true label for the current sample
    true_label = y_test.iloc[i]
    true_label_name = list(category_map.keys())[list(category_map.values()).index(true_label)]

    # Get the corresponding features from X_test
    sample_features = X_test.iloc[i].to_frame().T  # Convert to DataFrame with original feature names

    # Get the 'Name' of the sample from the original dataset (df)
    sample_name = df.loc[X_test.index[i], 'Name']

    # Predict the label using the ensemble model
    predicted_label = ensemble_model.predict(sample_features)
    predicted_label_name = list(category_map.keys())[list(category_map.values()).index(predicted_label[0])]

    # Determine if the prediction is correct
    correct_prediction = (true_label == predicted_label[0])

    # Append the result to the list
    results.append([sample_name, true_label_name, predicted_label_name, correct_prediction])

# Convert the results list to a DataFrame
results_df = pd.DataFrame(results, columns=['Name', 'True Label', 'Predicted Label', 'Correct Prediction'])

# Print the results as a table
print(results_df)