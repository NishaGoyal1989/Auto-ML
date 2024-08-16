from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import streamlit as st
import pandas as pd
from sklearn.metrics import precision_score,accuracy_score,recall_score,f1_score
from sklearn.linear_model import Ridge, Lasso, LogisticRegression

# Load the Boston housing dataset
@st.cache_resource
def grid_search_best_model(X_train, X_test, y_train, y_test, best_model_name, hyperparameters):
    # Define a dictionary of models with their respective hyperparameters for grid search
    models_params = {
        "Logistic Regressor": (LogisticRegression(), {}),
        "Ridge Classifier": (Ridge(), {'alpha': [0.1, 1.0, 10.0]}),
        "DecisionTree Classifier": (DecisionTreeClassifier(), {'max_depth': [None, 10, 20]}),
        "RandomForest Classifier": (RandomForestClassifier(), {'n_estimators': [100, 200, 300]}),
        "GradientBoosting Classifier": (GradientBoostingClassifier(), {'n_estimators': [100, 200, 300], 'learning_rate': [0.1, 0.05, 0.01]}),
        "XGBoost Classifier": (XGBClassifier(), {'n_estimators': [100, 200, 300], 'learning_rate': [0.1, 0.05, 0.01]}),
        "SVC": (SVC(), {'C': [0.1, 1.0, 10.0], 'kernel': ['linear', 'rbf']})
    }

    # Get the selected model and its hyperparameters
    model, params = models_params[best_model_name]

    # Update hyperparameters with user-provided values
    params.update(hyperparameters)

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(model, params, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best model found through grid search
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # Make predictions on the testing set using the best model
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    # Calculate mean squared error
    train_acc_score = accuracy_score(y_train, y_pred_train)
    test_acc_score = accuracy_score(y_test, y_pred_test)
    #st.write("Hi"+ str(best_model))
    return best_model, best_params, train_acc_score, test_acc_score

# @st.cache_resource
def classification_model(df,col_list,target):
    # Initialize a dictionary to store models and their scores
    # Initialize variables to track the best model, its MSE, and accuracy (not applicable for regression)
   
    best_model = None
    best_accuracy = 0  # Initialize with a high value

    # Define a dictionary to store MSE values for all models
    accuracy_values = {}

    X, y = df.drop(target,axis=1), df[target]
    X= X[col_list]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Initialize a dictionary to store models and their scores
    models = {
        "Logistic Regressor": LogisticRegression(),
        # "Ridge Classifier": Ridge(),
        "DecisionTree Classifier": DecisionTreeClassifier(),
        "RandomForest Classifier": RandomForestClassifier(),
        "GradientBoosting Classifier": GradientBoostingClassifier(),
        "XGBoost Classifier": XGBClassifier(),
        "SVC":SVC()
    }
    model_name=[]
    acc_score=[]
    f1score=[]
    recallscore=[]
    precisionscore=[]
    
    # Train and evaluate each model
    for name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
    
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate mean squared error
        acc_score1 = accuracy_score(y_test, y_pred_test)
        f1score1 = f1_score(y_test, y_pred_test)
        recallscore1 = recall_score(y_test, y_pred_test)
        precisionscore1 = precision_score(y_test, y_pred_test)
        
        # Update the best model if the current model has lower MSE
        if acc_score1> best_accuracy:
            best_accuracy=round(acc_score1,4)
            best_model = model
            best_model_name = name 
        
        # Store MSE value for current model
       
        model_name.append(name)
        acc_score.append(round(acc_score1,4))
        f1score.append(round(f1score1,4))
        recallscore.append(round(recallscore1,4))
        precisionscore.append(round(precisionscore1,4))
        
        # dict1['Model Name']= name
        # dict1['MSE'] = round(mse,4)
        # dict1['RMSE']=round(rmse,4)
        # dict1['R2 Score']=round(rsquared,4)
    
    # Return the best model along with its MSE value and MSE values for all models
    dict1= {
        "Model Name": model_name,
        "Accuracy Score": acc_score,
        "F1-Score": f1score,
        "Recall Score":recallscore,
        "Precision Score":precisionscore
    }
   # df1= pd.DataFrame.from_dict(dict1,orient= 'index')
    # df1= df1.T
    # df1.columns=['MSE','RMSE','MAPE','R2 Score']
    #df1.index.names=['Model Name']
    model_results= pd.DataFrame(dict1)
    model_results.sort_values(by='Accuracy Score', ascending=False, inplace=True)
    model_results.reset_index(drop=True, inplace=True)
        #best_model,mse= get_hyperparameters_from_user(X_train, X_test, y_train, y_test,dict1['best_model'])
    return model_results, best_model, best_model_name, round(accuracy_score(y_train, best_model.predict(X_train)), 4), best_accuracy

# @st.cache_resource     
def get_hyperparameters_from_user(df_t,col_list,target,selected_model):
   
    X, y = df_t.drop(target,axis=1), df_t[target]
    X= X[col_list]
    

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    hyperparameters = {}
    st.markdown("### Hyperparameters")
    if selected_model in ["Ridge Classifier", "Lasso Classifier"]:
        alpha = st.number_input("Alpha", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
        hyperparameters['alpha'] = [alpha]
    elif selected_model in ["DecisionTree Classifier"]:
        max_depth = st.number_input("Max Depth", min_value=1, max_value=100, value=10, step=1)
        hyperparameters['max_depth'] = [max_depth]
    elif selected_model in ["RandomForest Classifier"] :
        n_estimators = st.text_input("Number of Estimators use comma for mutiple values",100)
        criterion= st.multiselect("Select Criteria",["gini"],"gini")
        max_depth= st.text_input("Max Depth",10)
        min_samples_split= st.text_input("Min samples split",5)
        min_samples_leaf= st.text_input("Min samples leaf",2)
        hyperparameters['n_estimators'] = [int(x) for x in n_estimators.split(',')]
        hyperparameters['criterion'] = list(criterion)
        hyperparameters['max_depth'] = [int(x) for x in max_depth.split(',')]
        hyperparameters['min_samples_split'] = [int(x) for x in min_samples_split.split(',')]
        hyperparameters['min_samples_leaf'] = [int(x) for x in min_samples_leaf.split(',')]
        
    elif selected_model in ["GradientBoosting Classifier", "XGBoost Classifier"]:    #"RandomForest Classifier", 
        n_estimators = st.number_input("Number of Estimators", min_value=1, max_value=1000, value=100, step=1)
        learning_rate = st.number_input("Learning Rate", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
        hyperparameters['n_estimators'] = [n_estimators]
        hyperparameters['learning_rate'] = [learning_rate]
        
    elif selected_model == "SVC":
        C = st.number_input("C", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
        kernel = st.selectbox("Kernel", ["linear", "rbf"])
        hyperparameters['C'] = [C]

        hyperparameters['kernel'] = [kernel]
        
    best_model, best_params, train_score, test_score=grid_search_best_model(X_train,X_test,y_train,y_test,selected_model,hyperparameters)
    
    return best_model, best_params, train_score, test_score