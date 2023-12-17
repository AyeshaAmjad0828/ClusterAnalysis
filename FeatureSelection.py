import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os

# List of dataset file paths

dataset_files = [
    'Sources/WithLabels/HeartDisease_processed_features.csv',
    'Sources/WithLabels/BankMarketing_processed_features.csv',
    'Sources/WithLabels/HeartFailure_processed_features.csv',
    'Sources/WithoutLabels/CarEvaluation_processed_features.csv',
    'Sources/WithoutLabels/Mushroom_processed_features.csv'
    'Sources/WithoutLabels/OnlinePurchasingIntentions_processed_features.csv'
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
    return selected_features, model.feature_importances_

def variance_threshold(data, threshold=0.5):
    columns = data.columns  # Column names
    variances = data.var(axis=0)
    selected_features = columns[variances > threshold]
    filtered_data = data[selected_features]
    return filtered_data, variances    


# Create a folder to save the feature importance graph
output_folder = 'feature_importance_graphs'
os.makedirs(output_folder, exist_ok=True)
output_folder2 = 'feature_variance_graphs'
os.makedirs(output_folder2, exist_ok=True)




for dataset in dataset_files:
    # Load the current dataset
    data = pd.read_csv(dataset)
    if 'target' in data.columns:
        X = data.drop('target', axis=1)  # I have changed the header of dependent variable in each dataset to "target"
        y = data['target']

        # Perform feature selection
        selected_features, importances = random_forest_feature_importance(X, y, num_features=8)
 
        # Create an updated dataset with selected features
        X_selected = X[selected_features]
        updated_data = pd.concat([X_selected, y], axis=1)

        # Save the updated dataset to a new CSV file
        updated_dataset_file = dataset.replace('_features.csv', '_selected_features.csv')
        updated_data.to_csv(updated_dataset_file, index=False)
    
        print(f"Dataset with selected features saved to: {updated_dataset_file}")

        # Plot feature importances
        plt.figure()
        plt.bar(range(len(importances)), importances, align="center")
        plt.xticks(range(len(importances)), X.columns, rotation=45)
        plt.xlabel("Feature")
        plt.ylabel("Importance")
        plt.title("Feature Importances")

        # Save the plot as a PNG in the output folder
        output_filename = f"{os.path.splitext(os.path.basename(dataset))[0]}_feature_importance.png"
        output_path = os.path.join(output_folder, output_filename)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

        print(f"Feature importance graph saved to: {output_path}")

    else:
        X=data
        # Apply variance thresholding using the defined function
        thresholded_data, feature_variances = variance_threshold(X, threshold=0.05)

        # Save the updated dataset to a new CSV file
        updated_dataset_file = dataset.replace('_features.csv', '_selected_features.csv')
        thresholded_data.to_csv(updated_dataset_file, index=False)
    
        print(f"Dataset with selected features saved to: {updated_dataset_file}")

        # Create a bar graph of feature variances
        plt.figure(figsize=(8, 6))
        plt.bar(X.columns, feature_variances, color='orange')
        plt.title('Feature Variances')
        plt.xlabel('Features')
        plt.ylabel('Variance')
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.tight_layout()

        # Save the plot as a PNG in the output folder
        output_filename = f"{os.path.splitext(os.path.basename(dataset))[0]}_feature_variance.png"
        output_path = os.path.join(output_folder2, output_filename)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

        print(f"Feature variance graph saved to: {output_path}")

