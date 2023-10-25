import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# List of dataset file paths

dataset_files = [
    'ProcessedData/iris.csv',
    'ProcessedData/heartdisease.csv',
    'ProcessedData/wine.csv',
    'ProcessedData/carevaluation.csv',
    'ProcessedData/mushroom.csv',
    'ProcessedData/marketing.csv',
    'ProcessedData/onlineretail.csv',
    'ProcessedData/creditapproval.csv',
    'ProcessedData/purchasingintent.csv',
    'ProcessedData/heartfailure.csv'
]
dataset_files = [
    'dataset1_processed_features.csv',
    'dataset2_processed_features.csv',
    'dataset3_processed_features.csv',
    # Add more dataset file paths here
]

#Forward Selection Function
def forward_selection(X, y, model, metric=accuracy_score):
    features = []
    best_score = 0
    while True:
        remaining_features = list(set(X.columns) - set(features))
        if not remaining_features:
            break
        scores = []
        for feature in remaining_features:
            new_features = features + [feature]
            X_new = X[new_features]
            X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = metric(y_test, y_pred)
            scores.append((feature, score))
        best_feature, best_score = max(scores, key=lambda x: x[1])
        if best_score > best_score:
            best_score = best_score
            features.append(best_feature)
        else:
            break
    return features

# Random Forest Feature Importance Function
def random_forest_feature_importance(X, y, num_features=5):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    feature_importance = list(zip(X.columns, model.feature_importances_))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    selected_features = [feature for feature, _ in feature_importance[:num_features]]
    return selected_features

for dataset in dataset_files:
    # Load the current dataset
    data = pd.read_csv(dataset)
    X = data.drop('target', axis=1)  # Assuming 'target' is the target variable
    y = data['target']

    # Perform feature selection
    selected_features = random_forest_feature_importance(X, y, num_features=5)
    
    # Create an updated dataset with selected features
    X_selected = X[selected_features]
    updated_data = pd.concat([X_selected, y], axis=1)

    # Save the updated dataset to a new CSV file
    updated_dataset_file = dataset.replace('_features.csv', '_selected_features.csv')
    updated_data.to_csv(updated_dataset_file, index=False)
    
    print(f"Dataset with selected features saved to: {updated_dataset_file}")