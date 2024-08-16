from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import streamlit as st
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import os

@st.cache_data
def experiment_checker(exp_name=None, dataset_name=None, ml_type=None, target=None, task=None):
    if not dataset_name:
            return "Select the Dataset to train the model."
    elif not exp_name:
        return "Provide the Experiment Name."
    elif not ml_type:
            return "Select the type of ML Operation you want to perform."
    try:
        if not target:
                return "Select the target feature for training perpose."
    except:
        if not task:
                return "Select the type task you want to perform."   
    else:
        return False
    
# Load the Boston housing dataset
# @st.cache_resource
def grid_search_best_model(X_train, X_test, y_train, y_test, best_model_name, hyperparameters):
    # Define a dictionary of models with their respective hyperparameters for grid search
    models_params = {
        "Linear Regression": (LinearRegression(), {}),
        "Ridge Regression": (Ridge(), {'alpha': [0.1, 1.0, 10.0]}),
        "Lasso Regression": (Lasso(), {'alpha': [0.1, 1.0, 10.0]}),
        "ElasticNet": (ElasticNet(), {'alpha': [0.1, 1.0, 10.0], 'l1_ratio': [0.1, 0.5, 0.9]}),
        "Decision Tree Regression": (DecisionTreeRegressor(), {'max_depth': [None, 10, 20]}),
        "Random Forest Regression": (RandomForestRegressor(), {'n_estimators': [100, 200, 300]}),
        "Gradient Boosting Regression": (GradientBoostingRegressor(), {'n_estimators': [100, 200, 300], 'learning_rate': [0.1, 0.05, 0.01]}),
        "XGBoost Regression": (XGBRegressor(), {'n_estimators': [100, 200, 300], 'learning_rate': [0.1, 0.05, 0.01]}),
        "SVR": (SVR(), {'C': [0.1, 1.0, 10.0], 'kernel': ['linear', 'rbf']})
    }

    # Get the selected model and its hyperparameters
    model, params = models_params[best_model_name]

    # Update hyperparameters with user-provided values
    params.update(hyperparameters)

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(model, params, scoring='neg_mean_squared_error', cv=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best model found through grid search
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # Make predictions on the testing set using the best model
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    # Calculate mean squared error
    train_score = r2_score(y_train, y_pred_train)
    test_score = r2_score(y_test, y_pred_test)
    
    with open(os.path.join('Experiments', 'best_tuned_model.pkl'), 'wb') as f:
        pickle.dump(best_model, f) 
    return best_model, best_params, train_score, test_score

@st.cache_resource
def reg_model(df,col_list,target):
    # Initialize a dictionary to store models and their scores
    # Initialize variables to track the best model, its MSE, and accuracy (not applicable for regression)
    best_model = None
    best_model_name=None
    best_r2 = -1  # Initialize with a high value


    X, y = df.drop(target,axis=1), df[target]
    X= X[col_list]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize a dictionary to store models and their scores
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(),
        "ElasticNet": ElasticNet(),
        "Decision Tree Regression": DecisionTreeRegressor(),
        "Random Forest Regression": RandomForestRegressor(),
        "Gradient Boosting Regression": GradientBoostingRegressor(),
        "XGBoost Regression": XGBRegressor(),
        "SVR": SVR()
    }
    model_name=[]
    mse_l=[]
    rmse_l=[]
    mape_l=[]
    r2_score_l=[]
    # Train and evaluate each model
    for name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate mean squared error
        
        mse = mean_squared_error(y_test, y_pred_test)
        rmse = mean_squared_error(y_test, y_pred_test,squared= False)
        mape = mean_absolute_error(y_test, y_pred_test)
        rsquared = r2_score(y_test, y_pred_test) 
        # Update the best model if the current model has lower MSE
        if rsquared > best_r2:
            best_model_name = name
            best_model= model
            best_r2 = round(rsquared,4)
        
        # Store MSE value for current model
       
        model_name.append(name)
        mse_l.append(round(mse,4))
        rmse_l.append(round(rmse,4))
        mape_l.append(round(mape,4))
        r2_score_l.append(round(rsquared,4))
        
    # Return the best model along with its MSE value and MSE values for all models
    dict1= {
        "Model Name": model_name,
        "R2 Score":r2_score_l,
        "MSE": mse_l,
        "RMSE": rmse_l,
        "MAPE":mape_l        
    }
    
    model_results = pd.DataFrame(dict1)
    model_results.sort_values(by='R2 Score', ascending=False, inplace=True)
    model_results.reset_index(drop=True, inplace=True)
    
    return model_results, best_model, best_model_name, round(r2_score(y_train, best_model.predict(X_train))), round(best_r2,4)

# @st.cache_resource      
def get_hyperparameters_from_user(df_t,col_list,target,selected_model):
    X, y = df_t.drop(target,axis=1), df_t[target]
    X= X[col_list]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    hyperparameters = {}
    st.markdown("### Hyperparameters")
    if selected_model in ["Ridge Regression", "Lasso Regression", "ElasticNet"]:
        alpha = st.number_input("Alpha", min_value=0.001, max_value=10.0, value=1.0, step=0.001)
        hyperparameters['alpha'] = [alpha]
    elif selected_model in ["Decision Tree Regression"]:
        max_depth = st.number_input("Max Depth", min_value=1, max_value=100, value=10, step=1)
        hyperparameters['max_depth'] = [max_depth]
    elif selected_model in ["Random Forest Regression"] :
        n_estimators = st.text_input("Number of Estimators use comma for mutiple values",10)
        criterion= st.multiselect("Select Criteria",["squared_error", "absolute_error", "friedman_mse", "poisson"],"squared_error")
        max_depth= st.text_input("Max Depth",2)
        min_samples_split= st.text_input("Min samples split",2)
        min_samples_leaf= st.text_input("Min samples leaf",2)
        hyperparameters['n_estimators'] = [int(x) for x in n_estimators.split(',')]
        hyperparameters['criterion'] = list(criterion)
        hyperparameters['max_depth'] = [int(x) for x in max_depth.split(',')]
        hyperparameters['min_samples_split'] = [int(x) for x in min_samples_split.split(',')]
        hyperparameters['min_samples_leaf'] = [int(x) for x in min_samples_leaf.split(',')]
        
    elif selected_model in ["Random Forest Regression", "Gradient Boosting Regression", "XGBoost Regression"]:
        n_estimators = st.number_input("Number of Estimators", min_value=1, max_value=1000, value=100, step=1)
        learning_rate = st.number_input("Learning Rate", min_value=0.001, max_value=1.0, value=0.1, step=0.001)
        hyperparameters['n_estimators'] = [n_estimators]
        hyperparameters['learning_rate'] = [learning_rate]
    elif selected_model == "SVR":
        C = st.number_input("C", min_value=0.001, max_value=10.0, value=1.0, step=0.001)
        kernel = st.selectbox("Kernel", ["linear", "rbf"])
        hyperparameters['C'] = [C]
        hyperparameters['kernel'] = [kernel]
    best_model, params, train_score, test_score=grid_search_best_model(X_train,X_test,y_train,y_test,selected_model,hyperparameters)
    return best_model, params, train_score, test_score
    #return hyperparameters



