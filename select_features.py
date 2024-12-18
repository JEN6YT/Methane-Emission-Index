import pandas as pd
import numpy as np
import os
from zipfile import BadZipFile

# define models
from sklearn.model_selection import train_test_split

# LASSO
from sklearn.linear_model import LassoCV

def lasso(x_train_country, y_train_country):
    model = LassoCV(cv=5).fit(x_train_country, y_train_country)
    importance = model.coef_.tolist()
    return importance

def process_lasso(folder_path_country, file_names_country, feature, methane_df_t):
    lasso_df = pd.DataFrame(columns=feature)
    for file in file_names_country:
        file_path = os.path.join(folder_path_country, file)
        # print(file_path)
        try:
            # Attempt to read the Excel file
            df_country = pd.read_excel(file_path)
        except BadZipFile:
            print(f"Skipping file {file}: Not a valid zip file")
            continue
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue
        country_name = os.path.splitext(file)[0].split("_")[1]
        # print(country_name)
        # print(country_name)
        y_country = None
        for c in methane_df_t.columns:
            # print(c)
            if country_name in c:
                y_country = methane_df_t[c].values
                break
        # print(y_country)
        X_train, X_test, y_train, y_test = train_test_split(df_country, y_country, test_size=0.2, random_state=42)
        feature_rank = lasso(X_train, y_train)
        new_row_df = pd.Series(feature_rank, index=feature)
        new_row_df = new_row_df.to_frame().T
        lasso_df = pd.concat([lasso_df, new_row_df], axis = 0,ignore_index=True)

    lasso_feature_rank = abs(lasso_df.mean()).sort_values(ascending=False)
    lasso_feature = list(lasso_feature_rank.head(25).index)
    
    return lasso_feature



# Ridge

from sklearn.linear_model import RidgeCV

def process_ridge(folder_path_country, file_names_country, feature, methane_df_t):
    # Initialize DataFrame for Ridge results
    ridge_df = pd.DataFrame(columns=feature)

    for file in file_names_country:
        file_path = os.path.join(folder_path_country, file)
        try:
            # Attempt to read the Excel file
            df_country = pd.read_excel(file_path)
        except BadZipFile:
            print(f"Skipping file {file}: Not a valid zip file")
            continue
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue

        country_name = os.path.splitext(file)[0].split("_")[1]
        y_country = None
        for c in methane_df_t.columns:
            if country_name in c:
                y_country = methane_df_t[c].values
                break
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(df_country, y_country, test_size=0.2, random_state=42)

        # Fit Ridge regression model
        ridge_model = RidgeCV(alphas=[0.1, 1.0, 10.0], store_cv_values=True)  # Adjust alphas as needed
        ridge_model.fit(X_train, y_train)

        # Get the coefficients
        feature_rank = ridge_model.coef_

        # Convert to DataFrame and append
        new_row_df = pd.Series(feature_rank, index=feature)
        new_row_df = new_row_df.to_frame().T
        ridge_df = pd.concat([ridge_df, new_row_df], axis=0, ignore_index=True)

    ridge_feature_rank = abs(ridge_df.mean()).sort_values(ascending=False)
    ridge_feature = list(ridge_feature_rank.head(25).index)
    
    return ridge_feature



# SHAP Value

from sklearn.linear_model import Lasso
import shap

def process_shap(folder_path_country, file_names_country, feature, methane_df_t):

    # Initialize DataFrame for Shapley results
    shap_df = pd.DataFrame(columns=feature)

    for file in file_names_country:
        file_path = os.path.join(folder_path_country, file)
        try:
            # Attempt to read the Excel file
            df_country = pd.read_excel(file_path)
        except BadZipFile:
            print(f"Skipping file {file}: Not a valid zip file")
            continue
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue

        country_name = os.path.splitext(file)[0].split("_")[1]
        y_country = None
        for c in methane_df_t.columns:
            if country_name in c:
                y_country = methane_df_t[c].values
                break

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(df_country, y_country, test_size=0.2, random_state=42)

        X_train_100 = shap.utils.sample(X_train, 100)  # 100 instances for use as the background distribution

        # A simple linear model
        lasso = Lasso(alpha=0.1,max_iter=10000)
        lasso.fit(X_train, y_train)

        # Explain the model's predictions using SHAP
        explainer = shap.Explainer(lasso, X_train_100)
        shap_values = explainer(X_train) # (number of samples, number of features)

        # Compute shap value for each data point and take avg over the whole data set
        # number of avg = number of features

        shap_values_array = shap_values.values
        column_mean = np.mean(shap_values_array, axis=0)
        sorted_index = np.argsort(column_mean)
        sorted_features = X_train.columns[sorted_index] # feature names
        sorted_column_means = column_mean[sorted_index] # values
        new_row_df = pd.Series(sorted_column_means, index=sorted_features)
        new_row_df = new_row_df.to_frame().T
        shap_df = pd.concat([shap_df, new_row_df], ignore_index=True)

    shap_feature_rank = abs(shap_df.mean()).sort_values(ascending=False)
    shap_feature = list(shap_feature_rank.head(25).index)
    
    return shap_feature


# methane as y
methane_df = pd.read_csv(r"C:\Users\zhangy6\Downloads\CH4_raw_na.csv")

# Transform methane dataframe
for col in methane_df.columns[3:]:
    methane_df[col] = pd.to_numeric(methane_df[col], errors='coerce')
    num_df = methane_df.select_dtypes(include='number')
    medians = num_df.median()
    # Fill NaN values with the median of each numeric column
    methane_df[num_df.columns] = num_df.fillna(medians)

methane_df_last_31 = methane_df.iloc[:, -31:]
methane_df_3 = methane_df.iloc[:, 2]
methane_df = pd.concat([methane_df_3, methane_df_last_31], axis=1)

methane_df_t = methane_df.T
methane_df_t.columns = methane_df_t.iloc[0]
methane_df_t = methane_df_t.drop(methane_df_t.index[0])
# print(methane_df_t.head())

# for controllable features

sample = r"C:\epi_control\control_Afghanistan.xlsx"
sample_df = pd.read_excel(sample)
control_feature = list(sample_df.columns)

folder_path_country_control = r"C:\epi_control"
file_names_country_control = os.listdir(folder_path_country_control)
lasso_control_feature = process_lasso(folder_path_country=folder_path_country_control, file_names_country=file_names_country_control, feature=control_feature, methane_df_t=methane_df_t)
ridge_control_feature = process_ridge(folder_path_country=folder_path_country_control, file_names_country=file_names_country_control, feature=control_feature, methane_df_t=methane_df_t)
shap_control_feature = process_shap(folder_path_country=folder_path_country_control, file_names_country=file_names_country_control, feature=control_feature, methane_df_t=methane_df_t)

# for uncontrollable features

sample = r"C:\epi_uncontrol\uncontrol_Afghanistan.xlsx"
sample_df = pd.read_excel(sample)
uncontrol_feature = list(sample_df.columns)

folder_path_country_uncontrol = r"C:\epi_uncontrol"
file_names_country_uncontrol = os.listdir(folder_path_country_uncontrol)
lasso_uncontrol_feature = process_lasso(folder_path_country=folder_path_country_uncontrol, file_names_country=file_names_country_uncontrol, feature=uncontrol_feature, methane_df_t=methane_df_t)
ridge_uncontrol_feature = process_ridge(folder_path_country=folder_path_country_uncontrol, file_names_country=file_names_country_uncontrol, feature=uncontrol_feature, methane_df_t=methane_df_t)
shap_uncontrol_feature = process_shap(folder_path_country=folder_path_country_uncontrol, file_names_country=file_names_country_uncontrol, feature=uncontrol_feature, methane_df_t=methane_df_t)


# Added together

common_elements_control = set(lasso_control_feature) & set(ridge_control_feature) & set(shap_control_feature)
common_elements_uncontrol = set(lasso_uncontrol_feature) & set(ridge_uncontrol_feature) & set(shap_uncontrol_feature)


print(common_elements_control)
print(common_elements_uncontrol)