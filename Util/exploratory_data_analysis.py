import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy.stats import skew, kurtosis
import io

#########################################################################################
# Streamlit Session State Variables
if 'selected' not in st.session_state:
    st.session_state.selected = False
def select_box():
    st.session_state.selected = True
##########################################################################################

########################################################################################## 
# Automated EDA
# @st.cache_data
def automated_eda_report(dataframe):
    # Display basic information about the dataset
    st.subheader("Dataset Information")
    buffer = io.StringIO()
    dataframe.info(buf=buffer)
    lines = buffer.getvalue().splitlines()
    df = (pd.DataFrame([x.split() for x in lines[5:-2]], columns=lines[3].split()).drop(['Count', '#'],axis=1).rename(columns={'Non-Null':'Non-Null Count'}))
    
    col1, col2 = st.columns([0.3, 0.7])
    with col1:
        st.write(lines[1])
        st.write(lines[2])
        st.write(lines[-1])
        st.write(lines[-2])
    with col2:
        st.dataframe(df)
        
    # Display summary statistics of numerical columns
    st.subheader("Summary Statistics:")
    st.table(dataframe.describe())
    
    # Missing Values Analysis
    st.subheader("Missing Values Analysis:")
    missing_val_df = dataframe.isna().sum()
    missing_fig, missing_plot = plt.subplots(figsize=(8, int(len(dataframe.columns)/4)))
    sns.barplot(x=missing_val_df.values, y=missing_val_df.index, orient='h', ax=missing_plot)
    missing_plot.bar_label(missing_plot.containers[0], fontsize=6)
    missing_plot.set_xlabel("Missing Value Count", fontdict={'fontsize':8})
    missing_plot.set_ylabel("Featues", fontdict={'fontsize':8})
    missing_plot.tick_params(axis='x', labelsize=6, rotation=45)
    missing_plot.tick_params(axis='y', labelsize=6, rotation=45)
    st.pyplot(missing_fig)
    
    numerical_features = dataframe.select_dtypes(include=['float', 'int']).columns.tolist()
    
    # Correlation PLot
    st.subheader("Correlation:")
    corr_fig, corr_plot = plt.subplots(figsize=(int(len(dataframe.columns)), int(len(dataframe.columns)/2)))
    sns.heatmap(dataframe[numerical_features].corr(), cmap="coolwarm", annot=True, fmt=".3f")
    corr_plot.set_title("Correlation Matrix", fontsize=10)
    corr_plot.tick_params(axis='x', labelsize=6, rotation=45)
    corr_plot.tick_params(axis='y', labelsize=6, rotation=45)
    st.pyplot(corr_fig)

    # Distribution Plots & Boxplots for Numerical Features
    st.subheader("Distribution Analysis:")
    for i, feature in enumerate(numerical_features):
        st.write(f"**{i+1}. {feature}**")
        distribution_fig, (distribution_plot, box_plot) = plt.subplots(nrows=1, ncols=2, figsize=(8, int(len(dataframe.columns)/5)))
        
        # Distribution Plot
        sns.histplot(dataframe[feature], kde=True, bins=100, ax=distribution_plot)
        distribution_plot.set_title(f'Distribution of {feature}', fontsize=6)
        distribution_plot.set_xlabel(f"{feature}", fontsize=5)
        distribution_plot.set_ylabel("Count", fontsize=5)
        distribution_plot.tick_params(axis='x', labelsize=5, rotation=45)
        distribution_plot.tick_params(axis='y', labelsize=5, rotation=45)
        
        # Boxplot
        sns.boxplot(x=dataframe[feature], ax=box_plot)
        box_plot.set_title(f'Boxplot of {feature}', fontsize=6)
        box_plot.set_xlabel(f"{feature}", fontsize=5)
        box_plot.tick_params(axis='x', labelsize=5, rotation=45)
        box_plot.tick_params(axis='y', labelsize=5, rotation=45)
        
        plt.tight_layout()
        st.pyplot(distribution_fig)

        # Skewness for numerical features
        skewness = skew(dataframe[feature])
        st.write(f"Skewness={skewness:.2f}")

    # Count plots for categorical features
    categorical_features = dataframe.select_dtypes(include=['object']).columns
    st.subheader("Categorical Feature's Analysis:")
    for feature in categorical_features:
        if dataframe[feature].nunique() <= 50:
            count_fig, count_plot = plt.subplots(figsize=(8, int(len(dataframe.columns)/5)))
            sns.countplot(x=feature, data=dataframe, palette='viridis')
            for container in count_plot.containers:
                count_plot.bar_label(container, fmt='%d', fontsize=5)
            count_plot.set_title(f'Count plot of {feature}', fontsize=6)
            count_plot.set_xlabel(f"{feature}", fontsize=5)
            count_plot.set_ylabel("Count", fontsize=5)
            count_plot.tick_params(axis='x', labelsize=5, rotation=45)
            count_plot.tick_params(axis='y', labelsize=5, rotation=45)
            st.pyplot(count_fig)
###########################################################################################################

##########################################################################################################       
# Advanced (Bivariate) EDA 
# @st.cache_data(experimental_allow_widgets=True)
def advanced_eda(dataframe):
    
    numerical_features = dataframe.select_dtypes(include=['float', 'int']).columns.tolist()
    categorical_features = dataframe.select_dtypes(include=['object']).columns
    
    # Numerical vs Numerical
    st.subheader("Relationship Between Numerical Features:")
    for i, feature1 in enumerate(numerical_features):
        st.write(f"**{i+1}. {feature1}**")
        col1, col2 = st.columns([0.2, 0.8])
        with col1:
            feature2 = st.selectbox("Select the Feature", options=numerical_features, key=f"num_num_relation_{i}")
        with col2:
            # Scatter Plot
            num_num_relation_fig, num_num_relation_plot = plt.subplots(figsize=(8, int(len(dataframe.columns)/5)))
            sns.scatterplot(x=feature1, y=feature2, data=dataframe, palette='coolwarm')
            num_num_relation_plot.set_title(f'{feature1} vs {feature2}', fontsize=6)
            num_num_relation_plot.set_xlabel(f"{feature1}", fontsize=5)
            num_num_relation_plot.set_ylabel(f"{feature2}", fontsize=5)
            num_num_relation_plot.tick_params(axis='x', labelsize=5, rotation=45)
            num_num_relation_plot.tick_params(axis='y', labelsize=5, rotation=45)
            st.pyplot(num_num_relation_fig)
                
    # Categorical vs Numerical
    st.subheader("Relationship Between Categorical & Numerical Features:")
    i = 0
    for catg_feature in categorical_features:
        if dataframe[catg_feature].nunique() <= 50:  # Considering features with a small number of unique values
            st.write(f"**{i+1}. {catg_feature}**")
            i += 1
            col1, col2 = st.columns([0.2, 0.8])
            with col1:
                num_feature = st.selectbox("Select the Feature", options=numerical_features, key=f"catg_num_relation_{i}")
            with col2:
                # Barplot
                catg_num_relation_fig, catg_num_relation_plot = plt.subplots(figsize=(8, int(len(dataframe.columns)/5)))
                sns.barplot(x=catg_feature, y=num_feature, data=dataframe, palette='coolwarm', ci=False)
                for container in catg_num_relation_plot.containers:
                    catg_num_relation_plot.bar_label(container, fmt='%d', fontsize=5)
                catg_num_relation_plot.bar_label(catg_num_relation_plot.containers[0], fmt='%d', fontsize=5)
                catg_num_relation_plot.set_title(f'{catg_feature} vs {num_feature}', fontsize=6)
                catg_num_relation_plot.set_xlabel(f"{catg_feature}", fontsize=5)
                catg_num_relation_plot.set_ylabel(f"{num_feature}", fontsize=5)
                catg_num_relation_plot.tick_params(axis='x', labelsize=5, rotation=45)
                catg_num_relation_plot.tick_params(axis='y', labelsize=5, rotation=45)
                st.pyplot(catg_num_relation_fig)
#########################################################################################################

#########################################################################################################
# Dataset Summary
@st.cache_data
def get_dataset_summary(dataframe):
    buffer = io.StringIO()
    dataframe.info(buf=buffer)
    lines = buffer.getvalue().splitlines()
    df = (pd.DataFrame([x.split() for x in lines[5:-2]], columns=lines[3].split()).drop(['Count', '#'],axis=1).rename(columns={'Non-Null':'Non-Null Count'}))
    
    col1, col2 = st.columns([0.4, 0.6])
    with col1:
        st.title("")
        st.title("")
        st.title("")
        st.write(lines[1])
        st.write(lines[2])
        st.write(lines[-1])
        st.write(lines[-2])
    with col2:
        st.dataframe(df)
#####################################################################################################

#####################################################################################################
# Missing Value Analysis
@st.cache_data
def get_missing_val_analysis(dataframe):
    missing_val_df = dataframe.isna().sum()
    missing_fig, missing_plot = plt.subplots(figsize=(8, int(len(dataframe.columns)/4)))
    sns.barplot(x=missing_val_df.values, y=missing_val_df.index, orient='h', ax=missing_plot, ci=None)
    missing_plot.bar_label(missing_plot.containers[0])
    missing_plot.set_title("Missing Value Analysis", fontsize=8)
    missing_plot.set_xlabel("Missing Value Count", fontdict={'fontsize':8})
    missing_plot.set_ylabel("Featues", fontdict={'fontsize':8})
    missing_plot.tick_params(axis='x', labelsize=6, rotation=45)
    missing_plot.tick_params(axis='y', labelsize=6, rotation=45)
    st.pyplot(missing_fig)
####################################################################################################

#####################################################################################################
# Correlation matrics
@st.cache_data
def get_correlation_matrics(dataframe):
    
    numerical_features = dataframe.select_dtypes(include=['float', 'int']).columns.tolist()
    
    corr_fig, corr_plot = plt.subplots(figsize=(int(len(dataframe.columns)), int(len(dataframe.columns)/2)))
    sns.heatmap(dataframe[numerical_features].corr(), cmap="coolwarm", annot=True, fmt=".3f")
    corr_plot.set_title("Correlation Matrix", fontsize=10)
    corr_plot.tick_params(axis='x', labelsize=6, rotation=45)
    corr_plot.tick_params(axis='y', labelsize=6, rotation=45)
    st.pyplot(corr_fig)
###################################################################################################

####################################################################################################
# Discriptive Statistics
@st.cache_data
def get_discriptive_stats(dataframe):
    stats_df = dataframe.describe()
    stats_df = stats_df.reset_index().rename(columns={'index':'Statistics'})
    stats_df['Statistics'].replace({"count": "Count",
                                    "mean": "Mean",
                                    "std":"Standard Deviation",
                                    "min": "Minimum",
                                    '25%':"1st Quantile", 
                                    '50%': "2nd Quantile (Average)",
                                    '75%': "3rd Quantile",
                                    'max': "Maximum"}, inplace=True)        
    st.table(stats_df)
##################################################################################################

###################################################################################################
# Line PLot
@st.cache_data
def get_lineplot(dataframe, X, Y, hue_col, agg_method):
    # Grouping the dataset
    if agg_method != '':
        agg_method = agg_method
    else:
        agg_method = 'mean'
        
    if hue_col != '':
        hue_col = hue_col
    else:
        hue_col = None
    try :
        line_fig, line_plot = plt.subplots(figsize=(9, 3))
        sns.lineplot(data=dataframe, x=X, y=Y, hue=hue_col, palette='coolwarm', estimator=agg_method, markers=True)
        if hue_col:
            sns.move_legend(line_plot, "upper right", bbox_to_anchor=(1, 1), ncol=1, title=f'{hue_col}', frameon=True, fontsize=6)
        line_plot.set_title(f'{X} vs {Y}', fontsize=6)
        line_plot.set_xlabel(f"{X}", fontsize=5)
        line_plot.set_ylabel(f"{Y}", fontsize=5)
        line_plot.tick_params(axis='x', labelsize=5, rotation=45)
        line_plot.tick_params(axis='y', labelsize=5, rotation=45)
        st.pyplot(line_fig)
        return
    except:
        st.write(":red['Something Went Wrong. please try again.]")
        return
########################################################################################################################

###################################################################################################
# Scatter Plot
@st.cache_data
def get_scatterplot(dataframe, X, Y, hue_col, size_col):
    if size_col != '':
        size_col = size_col
    else:
        size_col = None
        
    if hue_col != '':
        hue_col = hue_col
    else:
        hue_col = None
    try :
        scatter_fig, scatter_plot = plt.subplots(figsize=(9, 3))
        sns.scatterplot(data=dataframe, x=X, y=Y, hue=hue_col, palette='coolwarm', size=size_col)
        if hue_col:
            sns.move_legend(scatter_plot, "upper right", bbox_to_anchor=(1, 1), ncol=1, title=f'{hue_col}', frameon=True, fontsize=6)
        scatter_plot.set_title(f'{X} vs {Y}', fontsize=6)
        scatter_plot.set_xlabel(f"{X}", fontsize=5)
        scatter_plot.set_ylabel(f"{Y}", fontsize=5)
        scatter_plot.tick_params(axis='x', labelsize=5, rotation=45)
        scatter_plot.tick_params(axis='y', labelsize=5, rotation=45)
        st.pyplot(scatter_fig)
        return
    except:
        st.write(":red['Something Went Wrong. please try again.]")
########################################################################################################################

###################################################################################################
# Bar Plot
@st.cache_data
def get_barplot(dataframe, X, Y, hue_col, agg_method):
    if agg_method != '':
        agg_method = agg_method
    else:
        agg_method = 'mean'
        
    if hue_col != '':
        hue_col = hue_col
    else:
        hue_col = None
    try :
        bar_fig, bar_plot = plt.subplots(figsize=(9, 3))
        sns.barplot(data=dataframe, x=X, y=Y, hue=hue_col, palette='coolwarm', ci=None, estimator=agg_method)
        if hue_col:
            sns.move_legend(bar_plot, "upper right", bbox_to_anchor=(1, 1), ncol=1, title=f'{hue_col}', frameon=True, fontsize=6)
        for container in bar_plot.containers:
            bar_plot.bar_label(container, fontsize=6)
        plt.suptitle(f'{X} vs {Y}', fontsize=8)
        bar_plot.set_xlabel(f"{X}", fontsize=6)
        bar_plot.set_ylabel(f"{Y}", fontsize=6)
        bar_plot.tick_params(axis='x', labelsize=6, rotation=45)
        bar_plot.tick_params(axis='y', labelsize=6, rotation=45)
        st.pyplot(bar_fig)
        return
    except:
        st.write(":red['Something Went Wrong. please try again.]")
########################################################################################################################

#######################################################################################################################
# Pie Plot
@st.cache_data
def get_pieplot(dataframe, Y):
    try:
        pie_fig, pie_ax = plt.subplots(figsize=(9, 3))
        data = dataframe[Y].value_counts().tolist()
        extrude_labels = np.zeros(len(data)).tolist()
        extrude_labels[data.index(max(data))] = 0.05
        pie_ax.pie(dataframe[Y].value_counts().values, labels=dataframe[Y].value_counts().index, explode=extrude_labels, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('coolwarm'))
        pie_ax.axis('equal')  # Equal aspect ratio ensures that the pie chart is circular.
        pie_ax.set_title(f'{Y} Distribution', fontsize=8)
        st.pyplot(pie_fig)
        return
    except:
        st.write(":red['Something Went Wrong. please try again.]")
################################################################################################################

###################################################################################################
# Bar Plot
@st.cache_data
def get_histplot(dataframe, X, hue_col, agg_method):
        
    if hue_col != '':
        hue_col = hue_col
    else:
        hue_col = None
        
    try :
        distribution_fig, distribution_plot = plt.subplots(figsize=(9, 3))
        sns.histplot(data=dataframe, x=X, hue=hue_col, kde=True, bins=100, stat=agg_method)
        if hue_col:
            sns.move_legend(distribution_plot, "upper right", bbox_to_anchor=(1, 1), ncol=1, title=f'{hue_col}', frameon=True, fontsize=6)
        distribution_plot.set_title(f'Distribution of {X}', fontsize=6)
        distribution_plot.set_xlabel(f"{X}", fontsize=5)
        distribution_plot.set_ylabel(f"{agg_method}", fontsize=5)
        distribution_plot.tick_params(axis='x', labelsize=5, rotation=45)
        distribution_plot.tick_params(axis='y', labelsize=5, rotation=45)
        st.pyplot(distribution_fig)
        return
    except:
        st.write(":red['Something Went Wrong. please try again.]")
########################################################################################################################

###################################################################################################
# Count Plot
@st.cache_data
def get_countplot(dataframe, X, hue_col, agg_method):
        
    if hue_col != '':
        hue_col = hue_col
    else:
        hue_col = None
        
    try :
        count_fig, count_plot = plt.subplots(figsize=(9,3))
        sns.countplot(x=X, data=dataframe, hue=hue_col, palette='viridis', stat=agg_method)
        for container in count_plot.containers:
            count_plot.bar_label(container, fmt='%d', fontsize=5)
        if hue_col:
            sns.move_legend(count_plot, "upper right", bbox_to_anchor=(1, 1), ncol=1, title=f'{hue_col}', frameon=True, fontsize=6)
        count_plot.set_title(f'Count plot of {X}', fontsize=6)
        count_plot.set_xlabel(f"{X}", fontsize=5)
        count_plot.set_ylabel(f"{agg_method}", fontsize=5)
        count_plot.tick_params(axis='x', labelsize=5, rotation=45)
        count_plot.tick_params(axis='y', labelsize=5, rotation=45)
        st.pyplot(count_fig)
        return
    except:
        st.write(":red['Something Went Wrong. please try again.]")
########################################################################################################################       
 
##########################################################################################################################
# HeatMap
@st.cache_data
def get_heatmap(dataframe, x_col, y_col, val_col, agg_func='mean'):
    if dataframe[val_col].dtype == 'object':
        st.write(f":red[{val_col} contains non-numerical data. Select column with numerical data OR you might want to choose a different visualization approach for categonon-numerical data.]")
        return
    else:
        try:
            if agg_func == '':
                agg_func = 'mean'
            # if groupby_col:
            #     temp_df = dataframe.groupby(groupby_col)[[x_col, y_col, val_col]].mean()
            #     pivot_df = temp_df.pivot_table(values=val_col, index=y_col, columns=x_col, aggfunc=agg_func, fill_value=0)
            # else:
            pivot_df = dataframe.pivot_table(values=val_col, index=y_col, columns=x_col, aggfunc=agg_func, fill_value=0)
                
            heatmap_fig, heatmap_plot = plt.subplots(figsize=(max(math.ceil(dataframe[x_col].nunique()/4), 2), max(math.ceil(dataframe[y_col].nunique()/4), 2)))
            sns.heatmap(pivot_df, annot=True, fmt=".1f", cmap="YlGnBu", annot_kws={'fontsize':5, 'rotation':45})
            heatmap_plot.set_title(f'{x_col} vs {y_col} by {val_col} ({agg_func})', fontsize=8)
            heatmap_plot.set_xlabel(x_col, fontsize=6)
            heatmap_plot.set_ylabel(y_col, fontsize=6)
            heatmap_plot.tick_params(axis='x', labelsize=5, rotation=45)
            heatmap_plot.tick_params(axis='y', labelsize=5, rotation=45)
            plt.tight_layout()
            st.pyplot(heatmap_fig)
            return
        except:
            st.write(":red['Something Went Wrong. please try again.]")
##############################################################################################################
