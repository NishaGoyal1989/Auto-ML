import statsmodels.api as sm
import streamlit as st
import pandas as pd
from Util import outlier as out 
from Util import missing_value as mv

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings("ignore")
import pickle

@st.cache_data
def encode_categorical_columns(df,column_cat_types):
    
    for _, row in column_cat_types.iterrows():            
        col = row['Column Name']            
        encoding_method = row['Encoding Method']
        
        if encoding_method == 'One-Hot Encoding':
            df = pd.get_dummies(df, columns=[col], prefix=[col])
        elif encoding_method == 'Label Encoding':
            df[col]=df[col].astype('category').cat.codes
        elif encoding_method == 'Ordinal Encoding':
            pass
        elif encoding_method == 'Frequency Encoding':
            freq_map = df[col].value_counts(normalize=True).to_dict()
            df[col] = df[col].map(freq_map)       
 
    return df
    #scaling(encoded_df,col4,target_column)
    #st.dataframe(encoded_df)
    #scaling(encoded_df,col4)

#def missing_value(df,col2,col3,col4,col_list,target_column):
@st.cache_data
def missing_value(df,column_data_types):

    for _, row in column_data_types.iterrows():            
        col = row['Column Name']            
        imputation_method = row['Imputation Method']
        if imputation_method == 'auto':
            if row['Data Type'] == 'object':
                df= mv.simple_imputation(df,col, 'most_frequent')
            else:
                df=mv.simple_imputation(df,col, 'mean')
        if imputation_method == 'mean' or imputation_method == 'median' or imputation_method == 'most_frequent':
            df=mv.simple_imputation(df,col, imputation_method)
        elif imputation_method == 'max':
            df= mv.max_imputation(df,col)
        elif imputation_method == 'min':
            df= mv.min_imputation(df,col)
        elif imputation_method == 'regression':  
            df= mv.linear_regression_imputation(df, col)                    
        elif imputation_method == 'classification':
            df= mv.logistic_regression_imputation(df, col)
        elif imputation_method == 'KNN':      
            df= mv.KNN_imputation(df, col)              

    return df 

@st.cache_data
def outlier_treatment(df,outliers_df):

    for _, row in outliers_df.iterrows():            
        col = row['Column Name']            
        treatment_method = row['Treatment Method']

        if treatment_method == 'auto': 
            df= out.auto_treatment(df,col)
        elif treatment_method == 'IQR':                        
            df= out.treat_and_visualize_outliers(df,col,'IQR')                            
        elif treatment_method == 'zscore': 
            df= out.treat_and_visualize_outliers(df,col,'z-score')     
        elif treatment_method == 'isolation_forest': 
            df= out.treat_and_visualize_outliers(df,col,'isolation_forest')                     
        elif treatment_method == 'svm': 
            df= out.treat_and_visualize_outliers(df,col,'svm')  
        elif treatment_method == 'knn': 
            df= out.treat_and_visualize_outliers(df,col,'knn')
            
    return df

@st.cache_data       
def col_include(df_original,col1,col2,col3,col4,target_column):
    col1.subheader("Feature Selection")
    df= df_original.copy()
    #
    df_data = pd.DataFrame({
        'Column_Name': df.columns,
        'Data_Type': df.dtypes.values
    })
    df_with_selections = df_data.copy()
    df_with_selections.insert(0, "Include", False)
    df_with_selections['Include'] = df_with_selections['Column_Name'] == target_column
    
    # Get dataframe row-selections from user with st.data_editor
    
    edited_df = col1.data_editor(
        df_with_selections, 
        hide_index=True,
        column_config={"Select": st.column_config.CheckboxColumn(required=True)},
        disabled=df.columns,
        num_rows="dynamic",
    )

    # Filter the dataframe using the temporary column, then drop the column
    selected_rows = edited_df[edited_df.Include]
    if (edited_df['Include']== 1).any():
        selected_rows.drop('Include', axis=1)
        col_list = selected_rows['Column_Name'].tolist() 
        dict1= missing_value(df,col2,col3,col4,col_list,target_column)
        #dict1 = encoding(df,col_list)  
        #return col_list
        return dict1

@st.cache_data
def scaling(df,X,y,scaler):
    
    #Standarization
    if scaler == 'Standarization':
        # Create a StandardScaler object
        scaler = StandardScaler()
        # Fit the scaler to your data and transform it
        scaled_data = scaler.fit_transform(df)
        scaled_data = pd.DataFrame(scaled_data, columns=df.columns)
    
    # Normalization
    elif scaler == 'Min-max Scaling':
        scaler = MinMaxScaler()
        scaled_X = scaler.fit_transform(X)
        scaled_data = pd.concat([pd.DataFrame(scaled_X,columns= X.columns),y],axis=1)
    
    # Robust Scaling   
    elif scaler == 'Robust Scaling':
        # Create a RobustScaler object  
        scaler = RobustScaler()
        # Fit the scaler to your data and transform it
        scaled_data = scaler.fit_transform(df)
        scaled_data = pd.DataFrame(scaled_data, columns=df.columns)
    
    return scaler, scaled_data
       
@st.cache_data
def backward_elimination(df_model,target_column):
    df = df_model.copy()
    # Assume 'X' is your feature matrix and 'y' is your target variable
    # Replace them with your actual feature matrix and target variable
    
    X = df.drop([target_column], axis=1) #'Unnamed: 0'
    
    y = df[target_column]

    # Add a constant column to the feature matrix (required for statsmodels)
    X = sm.add_constant(X)

    # Fit the initial model
    model = sm.OLS(y, X).fit()

    # Perform backward elimination
    while True:
        # Get the p-values for all features
        p_values = model.pvalues[1:]  # Exclude the constant term
        #st.write(p_values)
        # Find the feature with the highest p-value
        max_p_value = p_values.max()
        if max_p_value > 0.05:  # Set your significance level
            
            # Remove the feature with the highest p-value
            feature_to_remove = p_values.idxmax()
            X = X.drop(feature_to_remove, axis=1)
            
            # Fit the updated model
            model = sm.OLS(y, X).fit()
        else:
            break  # Exit the loop if no feature has a p-value greater than 0.05
    
    col_list = X.columns.tolist()[1:]
    my_dict = dict(zip(col_list,model.pvalues[1:]))
    
    return my_dict











