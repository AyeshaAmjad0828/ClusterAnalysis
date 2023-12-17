import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from openpyxl import Workbook
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel


# List of dataset file paths
dataset_files = [
    'Sources/WithLabels/HeartDisease.csv',
    'Sources/WithLabels/BankMarketing.csv',
    'Sources/WithLabels/HeartFailure.csv',
    'Sources/WithoutLabels/CarEvaluation.csv',
    'Sources/WithoutLabels/Mushroom.csv',
    'Sources/WithoutLabels/OnlinePurchasingIntentions.csv'
]


def clean_data(data):
    # 1.1 Drop sequential values columns
    sequential_columns = []
    for col in data.columns:
        if data[col].dtype in [np.int64, np.int32, np.float64]:
            differences = np.diff(data[col])
            if np.all(differences == differences[0]):
                sequential_columns.append(col)
    data_clean = data.drop(sequential_columns, axis=1)
    
    # 1.2 Impute missing values
    data_clean = data_clean.replace(r'^\s*$', pd.NA, regex=True)

    #identify numeric and categorical variables
    numeric_cols = data_clean.select_dtypes(include=np.number).columns
    categorical_cols = data_clean.select_dtypes(include=object).columns

    # Mean imputation for numeric fields and Mode imputation for categorical fields
    if len(categorical_cols) > 0:
        # Mode imputation for categorical fields
        data_clean[categorical_cols] = data_clean[categorical_cols].fillna(data_clean[categorical_cols].mode().iloc[0])
    data_clean[numeric_cols] = data_clean[numeric_cols].fillna(data_clean[numeric_cols].mean())
    
    
    return data_clean


def feature_encoding(data, encoding_option='onehot'):
    categorical_cols = data.select_dtypes(include=['object']).columns
    encoded_data = None  # Initialize encoded_data with a default value

    if len(categorical_cols) > 0:
        if encoding_option == 'onehot':
            encoded_data = pd.get_dummies(data, columns=categorical_cols)
        elif encoding_option == 'label':
            label_encoded_data = data.copy()
            for column in data.select_dtypes(include=['object']):
                label_encoded_data[column] = pd.factorize(data[column])[0]

            encoded_data = label_encoded_data
        else:
            raise ValueError("Invalid method. Use 'onehot' or 'label'.")
    else:
        encoded_data = data
    
    return encoded_data


def feature_scaling(data, scaling_type='Standardize'):
    if scaling_type == 'Min-Max':
        # 2.1 Min-Max Scaling
        scaler = MinMaxScaler()
    elif scaling_type == 'Standardize':
        # 2.2 Standardize Scaling
        scaler = StandardScaler()
    else:
        raise ValueError("Invalid scaling type!")
    
    data_scaled = scaler.fit_transform(data)
    
    return data_scaled

# def perform_eda(data):
#     # 3.1 Create histograms for numerical data
#     numerical_cols = data.select_dtypes(include='number').columns
#     fig, axes = plt.subplots(nrows=len(numerical_cols), ncols=1, figsize=(8, 4*len(numerical_cols)))

#     for i, col in enumerate(numerical_cols):
#         data[col].hist(ax=axes[i])
#         axes[i].set_title(col)
    
#     fig.tight_layout()
#     fig.savefig("EDA/numerical_histograms.png")  # Save histograms as .png image file
    

#     # 3.2 Create bar graph for categorical/string data
#     categorical_cols = data.select_dtypes(include=['object']).columns
#     if len(categorical_cols) > 0:
#         for i, col in enumerate(categorical_cols):
#             data[col].value_counts().plot(kind='bar', ax=axes[i])
#             axes[i].set_title(col)
    
#         fig.tight_layout()
#         fig.savefig("EDA/categorical_bargraphs.png")  # Save bar graphs as .png image file
#     else:
#         print("no categorical variables found in the dataset")
    
#     frequency_tables = {}
#     for col in categorical_cols:
#         freq_table = data[col].value_counts()
#         freq_table.columns = [col, 'Frequency']
#         frequency_tables[col] = freq_table

#     writer = pd.ExcelWriter('EDA/frequency_tables.xlsx', engine='openpyxl')
#     for feature, table in frequency_tables.items():
#         table.to_excel(writer, sheet_name=feature, index=False)

#     writer.save()
#     writer.close()





for dataset in dataset_files:
    # Load dataset
    data = pd.read_csv(dataset)
    if 'target' in data.columns:
        X = data.drop('target', axis=1)  # I have changed the header of dependent variable in each dataset to "target"
        y = data['target']

    else:
        X=data
    
    # Data cleaning
    data_cleaned = clean_data(X)

    # Feature encoding
    data_encoded = feature_encoding(data_cleaned,'onehot')    

    # Feature scaling
    column_names = data_encoded.columns.tolist() 
    scaling_type = 'Min-Max'  # Adjust the scaling type as per your requirement
    data_scaled = feature_scaling(data_encoded, scaling_type)
    data_scaled = pd.DataFrame(data_scaled, columns=column_names)

    # # Perform EDA
    if 'target' in data.columns:
        ProcessedData = pd.concat([data_scaled, y], axis=1)
    else:
        ProcessedData = data_scaled
        

    # perform_eda(ProcessedData)

    # Save the updated dataset to a new CSV file
    updated_dataset_file = dataset.replace('.csv', '_processed_features.csv')
    ProcessedData.to_csv(updated_dataset_file, index=False)
    
    print(f"Dataset with processed features saved to: {updated_dataset_file}")
    