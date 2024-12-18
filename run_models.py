# control_features = ['MPE', 'NXA', 'TCG', 'TKP', 'MKP', 'BTZ', 'HFD_cbrt_diff', 'PFL']
# uncontrol_features = ['SHI', 'CBP', 'FCD', 'BCA_cbrt_diff', 'LUF', 'OEB', 'PAE_cbrt_diff', 'FCL']

control_features = ['MPE', 'NXA', 'TCG', 'TKP', 'MKP', 'BTZ', 'HFD', 'PFL']
uncontrol_features = ['SHI', 'CBP', 'FCD', 'BCA', 'LUF', 'OEB', 'PAE', 'FCL']

import optuna
import xgboost as xgb
import pandas as pd
import os
from zipfile import BadZipFile

# save the featured data

def select_features(folder_path_country, file_names_country, feature, file_path_new):
    os.makedirs(file_path_new, exist_ok=True)
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

        x_country = pd.DataFrame(columns=feature)
        x = None
        for f in feature:
            for c in df_country.columns:
                if f in c:
                    x = df_country[c].values
                    # add x to x_country
                    x_country[f] = x
                    break
        file_path_new_path = os.path.join(file_path_new, f"{country_name}.xlsx")
        x_country.to_excel(file_path_new_path, index=False)
        print("Done")

# control
folder_path_country_control = r"C:\epi_control"
file_names_country_control = os.listdir(folder_path_country_control)
new_filepath_control = r"C:\epi_control_features"
# select_features(folder_path_country=folder_path_country_control,file_names_country=file_names_country_control, feature=control_features, file_path_new=new_filepath_control)

# uncontrol 
folder_path_country_uncontrol = r"C:\epi_uncontrol"
file_names_country_uncontrol = os.listdir(folder_path_country_uncontrol)
new_filepath_uncontrol = r"C:\epi_uncontrol_features"
# select_features(folder_path_country=folder_path_country_uncontrol,file_names_country=file_names_country_uncontrol, feature=uncontrol_features, file_path_new=new_filepath_uncontrol)




# Define the objective function for Optuna

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

def xgb_model(folder_path_country, file_names_country, methane_df_tt):
    xgb_results = []

    error_df = pd.DataFrame(columns=['Year', 'Country', 'Error Percentage'])


    # for each country
    for file in file_names_country:
        file_path = os.path.join(folder_path_country, file)
        try:
            # Attempt to read the Excel file
            X = pd.read_excel(file_path, engine='openpyxl')
            if X.empty:
                continue
            # X = X.values
        except BadZipFile:
            print(f"Skipping file {file}: Not a valid zip file")
            continue
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue
        country_name = os.path.splitext(file)[0]

        Y = None
        for c in methane_df_tt.columns:
            if country_name in c:
                Y = methane_df_tt[c].values
                # print(type(Y))
                break

        # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
        X_train = X.iloc[:25, :].values
        # print(X_train.shape)
        y_train = Y[:25]
        # print(y_train.shape)
        X_test = X.iloc[25:, :].values
        # print(X_test.shape)
        y_test = Y[25:]
        # print(y_test.shape)
        # Standardize the features
        scaler_X = StandardScaler()
        X_train = scaler_X.fit_transform(X_train)
        X_test = scaler_X.transform(X_test)

        # Standardize the target variable based on the training data
        scaler_y = StandardScaler()
        y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test = scaler_y.transform(y_test.reshape(-1, 1)).ravel()  # Standardize y_test for evaluation

        def xgb_objective(trial):
            # Define hyperparameters to tune
            param = {
                'objective': 'reg:squarederror',
                'booster': 'gbtree',
                'lambda': trial.suggest_loguniform('lambda', 1e-8, 1.0),
                'alpha': trial.suggest_loguniform('alpha', 1e-8, 1.0),
                'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.5, 0.7, 0.9, 1.0]),
                'subsample': trial.suggest_categorical('subsample', [0.5, 0.7, 0.9, 1.0]),
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 9),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
            }

            # Train the model
            model = xgb.XGBRegressor(**param)
            model.fit(X_train, y_train)

            # Predict and calculate RMSE
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)

            return mse

        # Set up Optuna study
        study = optuna.create_study(direction='minimize')
        study.optimize(xgb_objective, n_trials=50, timeout=600)  # Adjust trials and timeout as needed

        # Train final model with best parameters
        best_params = study.best_trial.params
        model = xgb.XGBRegressor(**best_params)
        model.fit(X_train, y_train)

        # Predict and calculate metrics
        y_pred = model.predict(X_test)

        # Create a graph for each test date
        for i in range(len(y_test)):
            error_df.loc[error_df.shape[0]] = [i + 2017, country_name, (y_pred[i] - y_test[i]) / y_pred[i]]
            print(f"Year: {i + 2017}, Country: {country_name}, Error Percentage: {(y_pred[i] - y_test[i]) / y_test[i]}")

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Store the results for this country
        xgb_results.append({
            'Country': country_name,
            'MSE': mse,
            'R2': r2
        })

    # Create a DataFrame from the results
    xgb_results_df = pd.DataFrame(xgb_results)

    # Calculate the average MSE and R² score across all countries
    xgb_average_mse = xgb_results_df['MSE'].mean()
    xgb_average_r2 = xgb_results_df['R2'].mean()

    return xgb_average_mse, xgb_average_r2, error_df


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


# file_names_country_feature_control = os.listdir(new_filepath_control)
# xgb_average_mse_control, xgb_average_r2_control, error_control = xgb_model(folder_path_country=new_filepath_control, file_names_country=file_names_country_feature_control, methane_df_tt=methane_df_t)
# print(xgb_average_mse_control) 
# print(xgb_average_r2_control)
# error_control.to_excel(r"C:\Users\zhangy6\github\Methane-Emission-Index\control.xlsx")
file_names_country_feature_uncontrol = os.listdir(new_filepath_uncontrol)
xgb_average_mse_uncontrol, xgb_average_r2_uncontrol, error_uncontrol = xgb_model(folder_path_country=new_filepath_uncontrol, file_names_country=file_names_country_feature_uncontrol, methane_df_tt=methane_df_t)
print(xgb_average_mse_uncontrol)
print(xgb_average_r2_uncontrol)
error_uncontrol.to_excel(r"C:\Users\zhangy6\github\Methane-Emission-Index\uncontrol.xlsx")