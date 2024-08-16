import pandas as pd
import streamlit as st 
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

# Function to get columns and their data types
@st.cache_data
def get_column_data_types(df_mv):
    # Exclude 'ADDRESS' column from the DataFrame
    # columns_to_exclude = ['ADDRESS']
    columns_with_null = df_mv.columns[df_mv.isnull().any()].tolist()
    
    # Remove 'ADDRESS' from columns_with_null
    # columns_with_null = [col for col in columns_with_null if col not in columns_to_exclude]

    return df_mv[columns_with_null].dtypes.reset_index().rename(columns={0: 'Data Type', 'index': 'Column Name'})

# Function to perform imputation using SimpleImputer
@st.cache_data
def simple_imputation(df_mv,column, strategy):
    if df_mv[column].isnull().any(): 
        missing_mask_index = df_mv[df_mv[column].isnull()].index
        imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
        df_mv[column] = imputer.fit_transform(df_mv[[column]].values.reshape(-1, 1)).reshape(-1)
        #st.dataframe(df_mv[df_mv.index.isin(missing_mask_index)]) 
        return df_mv
    else:
        st.write("No null values")

# Function to perform Mean imputation
@st.cache_data
def min_imputation(df_mv,column):
    if df_mv[column].isnull().any(): 
        # Save the index of rows with missing values in the specified column
        missing_mask_index = df_mv[df_mv[column].isnull()].index
        min_value = df_mv[column].min()
        df_mv[column].fillna(min_value, inplace=True)
        return df_mv
        #st.dataframe(df_mv[df_mv.index.isin(missing_mask_index)])
    else:
        st.write("No null values")
  
# Function to perform Max imputation
@st.cache_data
def max_imputation(df_mv,column):
    if df_mv[column].isnull().any(): 
        missing_mask_index = df_mv[df_mv[column].isnull()].index
        max_value = df_mv[column].max()
        df_mv[column].fillna(max_value, inplace=True)
        return df_mv
        #st.dataframe(df_mv[df_mv.index.isin(missing_mask_index)])
    else:
        st.write("No null values")  

@st.cache_data
def logistic_regression_imputation(df_mv, target_column):
    if df_mv[target_column].isnull().any():
        df_copy= df_mv.copy()
        #df_copy= df_copy.drop(['ADDRESS'],axis=1)
        # Identify features and target variable
        # Identify numerical and categorical columns
        numerical_columns = df_copy.select_dtypes(include=['float64', 'int64']).columns
        categorical_columns = df_copy.select_dtypes(include=['object']).columns.difference([target_column])

        # Fill missing values with mean for numerical columns
        df_copy[numerical_columns] = df_copy[numerical_columns].fillna(df_copy[numerical_columns].mean())

        # Fill missing values with mode for categorical columns
        df_copy[categorical_columns] = df_copy[categorical_columns].apply(lambda x: x.fillna(x.mode().iloc[0]))
        target = df_copy[target_column]

        # Identify missing values in the target column
        missing_mask = target.isnull()

        # Split the data into two parts: one with missing values and one withoutlit 
        complete_data = df_copy[~missing_mask]
        missing_data = df_copy[missing_mask]
        features = complete_data.drop(columns=[target_column])
        missing_data = missing_data.drop(columns=[target_column])

        # Drop rows with any missing values in features
        complete_data = complete_data.dropna(subset=features.columns[features.isnull().any()])
        features = complete_data.drop(columns=[target_column])
        # Encode categorical columns
        le = LabelEncoder()
        for column in features.select_dtypes(include=['object']).columns:
            features[column] = le.fit_transform(features[column])
        for column in missing_data.select_dtypes(include=['object']).columns:
            missing_data[column] = le.fit_transform(missing_data[column])

        # Encode the target variable
        target_encoder = LabelEncoder()
        complete_data[target_column] = target_encoder.fit_transform(complete_data[target_column])

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            complete_data[target_column],
            test_size=0.2,
            random_state=42
        )

        # Train a logistic regression model
        model = LogisticRegression(multi_class='ovr')
        model.fit(X_train, y_train)

        # Predict the missing values
        predicted_values = model.predict(missing_data)

        # Inverse transform the predicted values to get original categorical labels
        predicted_labels = target_encoder.inverse_transform(predicted_values)
        # Use the trained model to predict missing values
        imputed_index = df_copy[df_copy[target_column].isnull()].index
        # Update the missing values in the original dataframe
        df_mv.loc[imputed_index, target_column] = predicted_labels
        return df_mv
        # Display only the imputed rows with highlighted values
        #st.dataframe(df_mv[df_mv.index.isin(imputed_index)])
    else:
        st.write("No null values")    

@st.cache_data
def linear_regression_imputation(df_mv, target_column):
    if df_mv[target_column].isnull().any():
        df_copy = df_mv.copy()  # Create a copy of the original dataframe
        #df_copy= df_copy.drop(['ADDRESS'],axis=1)
        # Identify numerical and categorical columns
        numerical_columns = df_copy.select_dtypes(include=['float64', 'int64']).columns.difference([target_column])
        categorical_columns = df_copy.select_dtypes(include=['object']).columns

        # Fill missing values with mean for numerical columns
        df_copy[numerical_columns] = df_copy[numerical_columns].fillna(df_copy[numerical_columns].mean())

        # Fill missing values with mode for categorical columns
        df_copy[categorical_columns] = df_copy[categorical_columns].apply(lambda x: x.fillna(x.mode().iloc[0]))

        # Perform one-hot encoding if categorical columns are present
        df_encoded = pd.get_dummies(df_copy, columns=categorical_columns, drop_first=True)

        # Split the data into two parts: one with missing values and one without
        missing_data = df_encoded[df_copy[target_column].isnull()]
        complete_data = df_encoded.dropna(subset=[target_column])

        # Create features (X) and target variable (y) for the complete data
        X_train = complete_data.drop(columns=[target_column])
        y_train = complete_data[target_column]

        # Identify features and target variable for missing data
        X_missing = missing_data.drop(columns=[target_column])

        # Initialize a linear regression model
        model = LinearRegression()

        # Train the model
        model.fit(X_train, y_train)

        # Use the trained model to predict missing values
        imputed_index = df_copy[df_copy[target_column].isnull()].index
        df_mv.loc[imputed_index, target_column] = model.predict(X_missing)
        return df_mv
        # Display only the imputed rows with highlighted values
        #st.dataframe(df_mv[df_mv.index.isin(imputed_index)])
    
     
    else:
        st.write("No null values")

@st.cache_data
def KNN_imputation(df_mv, target_column, n_neighbors=3):
    if df_mv[target_column].isnull().any():
        df_copy= df_mv.copy()
        #df_copy.drop(['ADDRESS'], axis=1, inplace=True)
        # Identify numerical and categorical columns
        numerical_columns = df_copy.select_dtypes(include=['float64', 'int64']).columns.difference([target_column])
        categorical_columns = df_copy.select_dtypes(include=['object']).columns

        # Fill missing values with mean for numerical columns
        df_copy[numerical_columns] = df_copy[numerical_columns].fillna(df_copy[numerical_columns].mean())

        # Fill missing values with mode for categorical columns
        df_copy[categorical_columns] = df_copy[categorical_columns].apply(lambda x: x.fillna(x.mode().iloc[0]))

        # Identify categorical columns
        categorical_columns = df_copy.select_dtypes(include=['object']).columns

        # One-hot encode categorical columns
        df_encoded = pd.get_dummies(df_copy, columns=categorical_columns, drop_first=True)

        # Fill missing values in the target column with a placeholder (e.g., '#')
        df_encoded[target_column].fillna(-999, inplace=True)

        # Initialize a KNN imputer
        imputer = KNNImputer(missing_values=-999, n_neighbors=n_neighbors)

        # Impute missing values in the entire dataframe
        imputed_data = imputer.fit_transform(df_encoded)

        # Create a new dataframe with imputed values
        df_imputed = pd.DataFrame(data=imputed_data, columns=df_encoded.columns)
        
        # Use the trained model to predict missing values
        imputed_index = df_copy[df_copy[target_column].isnull()].index
        
        df_mv.loc[imputed_index, target_column] = df_imputed.loc[imputed_index, target_column].astype(df_mv[target_column].dtype)
        return df_mv
       # Display only the imputed rows with highlighted values
        #st.dataframe(df_mv[df_mv.index.isin(imputed_index)])
     
    else:
        st.write("No null values")    

