import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import shapiro
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")


    
def detect_outliers(df, method='iqr', threshold=1.5):
    
    columns = df.select_dtypes(include=['float64', 'int64']).columns

    outlier_columns = []

    for col in columns:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - (threshold * IQR)
            upper_bound = Q3 +( threshold * IQR)
            outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
        else:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers = z_scores > threshold

        if outliers.any():
            outlier_columns.append(col)

    return outlier_columns

def auto_treatment(df,col):
    
    stat, p_value = shapiro(df[col])
    
    if p_value > 0.05 :
        
        df = treat_and_visualize_outliers(df,col,'z-score')
    else:
        df = treat_and_visualize_outliers(df,col,'IQR')
    return df

# Function to treat outliers and visualize boxplots
def treat_and_visualize_outliers(df,column, method, threshold=1.5):
    
# Function to treat outliers
    def treat_outliers(df,column, method, threshold):
              
        if method == 'z-score':
            # Z-score method for outlier detection
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            outliers = z_scores > threshold
            # Replace outliers with the median value
            df.loc[outliers, column] = df[column].median()
        elif method =='svm':
            kernel='rbf' 
            nu=0.05
            mean_value= df[column].mean()
            df[column]= df[column].fillna(mean_value)
            # Extract the column data
            column_data =df[column].values.reshape(-1, 1)

            # Initialize OneClassSVM model
            model = OneClassSVM(kernel=kernel, nu=nu)

            # Fit the model
            model.fit(column_data)

            # Predict outliers (inliers will be labeled as 1, outliers as -1)
            outliers = model.predict(column_data) == -1
            outliers = pd.Series(outliers)
        elif method== 'knn':
            mean_value= df[column].mean()
            df[column]= df[column].fillna(mean_value)
            # Extract the column data
            column_data = df[column].values.reshape(-1, 1)

            # Initialize kNN model
            knn = NearestNeighbors(n_neighbors=5)
            knn.fit(column_data)

            # Compute distances to k-nearest neighbors for each observation
            distances, indices = knn.kneighbors(column_data)

            # Identify outliers based on distances
            outliers = distances[:, -1] > distances[:, -2]
            outliers = pd.Series(outliers)
        elif method == 'isolation_forest':
            contamination=0.05
            random_state=42
            mean_value= df[column].mean()
            df[column]= df[column].fillna(mean_value)
            # Extract the column data
            column_data = df[column].values.reshape(-1, 1)

            # Initialize Isolation Forest model
            model = IsolationForest(contamination=contamination, random_state=random_state)

            # Fit the model
            model.fit(column_data)

            # Predict outliers (inliers will be labeled as 1, outliers as -1)
            outliers = model.predict(column_data) == -1
            outliers = pd.Series(outliers)
        
            
        elif method == 'IQR':
            # Interquartile Range (IQR) method for outlier detection
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            outliers = (df[column] < Q1 - 1.5 * IQR) | (df[column] > Q3 + 1.5 * IQR)
            #treated_df.loc[outliers, column] = df[column].median() 
            # Treat outliers by replacing them with the median value
            
            # Replace values outside the specified range
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1    
        # Cap outliers at the 10th and 90th percentiles
        lower_limit = Q1-(1.5*IQR)
        upper_limit = Q3+(1.5*IQR)
        df.loc[outliers,column] = df.loc[outliers,column].apply(lambda x: lower_limit if x < lower_limit else (upper_limit if x > upper_limit else x)) 
        
        return df   

    # Treat outliers in the specific column and get treated rows
    treated_rows_df = treat_outliers(df, column, method=method, threshold=threshold)
    
    return treated_rows_df


        
        

