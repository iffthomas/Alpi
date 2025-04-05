import pandas as pd
import numpy as np
from os.path import join
import os
import csv
from sklearn.model_selection import GridSearchCV, PredefinedSplit, TimeSeriesSplit
from xgboost import XGBRegressor
# depending on your IDE, you might need to add datathon_eth. in front of data
from data import DataLoader, SimpleEncoding, log_results, ImputationEncoding, FeatureEncoding
# depending on your IDE, you might need to add datathon_eth. in front of forecast_models
from forecast_models import SimpleModel, ARIMAModel, GaussianProcessModel, XGBoostModel, GAMModel, CatBoostModel, MultiTimeSeriesForecaster, LightGBMModel
import warnings
warnings.filterwarnings("ignore")
def main(zone: str, encoding_name: str, model_name: str, train_test: bool, split_date :str , feature_sets:str, global_model: bool, inputation_method:str = "ffill", start_date:str = None):
    """

    Train and evaluate the models for IT and ES
    encoding_name: name of the encoding used for logging
    model_name name of the model used for logging

    """

    # Inputs
    input_path = r"datasets2025"
    output_path = r"outputs"

    # Load Datasets
    loader = DataLoader(input_path)
    # features are holidays and temperature

    if train_test == True:
        print(f"Train_testing will be done with {split_date} ")
        training_set, test_set, train_features, test_features = loader.load_data_train_test(country = country, 
                                                                                            split_date=split_date
                                                                                            )
        
        #join train_features and test_features
        features_train_and_test = pd.concat([train_features,test_features])

        print(features_train_and_test)

        #calculate a lot of features for the training set



    else:
        training_set, features, test_set = loader.load_data(zone)
        print("Predicting For the Dummy-Set")

    """
    EVERYTHING STARTING FROM HERE CAN BE MODIFIED.
    """
    team_name = "LehmanBrothers"
    # Data Manipulation and Training
    start_training = training_set.index.min()

    if start_date is not None:
        #find the index of the start date in the training set
        start_training = training_set.index.get_loc(start_date)
        #get the date of the index
        start_training = training_set.index[start_training]
        print(start_training)

    end_training = training_set.index.max()
    print(end_training)
    start_forecast, end_forecast = test_set.index[0], test_set.index[-1]
    range_forecast = pd.date_range(start=start_forecast, end=end_forecast, freq="1H")

    forecast = pd.DataFrame(columns=training_set.columns, index=range_forecast)


    if global_model:
        print("We are using a global model")
        print("------------------------------------------------------")

        if train_test:
            pass

    else:
        print("We are using Models for each customer")
        print("------------------------------------------------------")
        for costumer in training_set.columns.values:

            print(f"Customer: {costumer}")
            consumption = training_set.loc[:, costumer]
            if train_test:

                feature_dummy =  features_train_and_test[feature_sets].loc[start_training:]
                if encoding_name == "calculate_custom_features":
                    feature_dummy =  features_train_and_test.loc[start_training:]
            else:
                feature_dummy = features[feature_sets].loc[start_training:]

            
            if encoding_name == "baseline_encoding":

                encoding = SimpleEncoding(
                    consumption, feature_dummy, end_training, start_forecast, end_forecast
                )

                feature_past, feature_future, consumption_clean = (
                encoding.meta_encoding()
            )
            elif encoding_name == "custom_encoding":
                encoding = ImputationEncoding(
                    consumption, feature_dummy, start_training, end_training, start_forecast, end_forecast, method=inputation_method
                )
                feature_past, feature_future, consumption_clean, consumption_test= (
                    encoding.meta_encoding()
                )
        
            elif encoding_name=="calculate_custom_features":

                encoding = FeatureEncoding(
                    consumption, feature_dummy, start_training, end_training, start_forecast, end_forecast, customer=costumer
                )

                feature_past, feature_future, consumption_clean, consumption_test = encoding.calculate_features()

            else:
                print("NO Encoding specified sofar")
                raise ValueError("No encoding specified")

            if model_name == "linear_regression":
                model = SimpleModel()
                model.train(feature_past, consumption_clean)

                output = model.predict(feature_future)

            elif model_name == "arima":

                model = ARIMAModel(order=(1, 1, 1))
                model.train(consumption_clean)

                n = len(consumption_clean)
                forecast_horizon = len(test_set)  # or any desired horizon
                output = model.predict(start=n, end=n + forecast_horizon - 1)

                #if the output contains NaN values, fill them with the mean of the training set
                if np.isnan(output).any():
                    print("NaN values found in the output. Filling with mean.")
                    output = output.fillna(consumption_clean.mean())

            elif model_name == "random_forest":
                # Random Forest Model
                model = RandomForestModel()
                model.train(feature_past, consumption_clean)
                output = model.predict(feature_future)

            elif model_name == "gaussian_process":
                # Gaussian Process Model
                model = GaussianProcessModel()
                model.train(feature_past, consumption_clean)
                output = model.predict(feature_future)

            elif model_name == "xgboost":

                model = XGBoostModel()
                model.train(feature_past, consumption_clean)
                output = model.predict(feature_future)

            elif model_name == "gam":
                # Generalized Additive Model (GAM)
                model = GAMModel()
                model.train(feature_past, consumption_clean)
                output = model.predict(feature_future)

            elif model_name == "catboost":
                # CatBoost Model
                try:
                    model = CatBoostModel()
                    model.train(feature_past, consumption_clean)
                    output = model.predict(feature_future)
                except Exception as e:
                    custom_col = [col for col in feature_past.columns if costumer in col]
                    output = feature_future[custom_col]

            elif model_name == "ligthgbm":
                # LightGBM Model
                model = LightGBMModel()
                model.train(feature_past, consumption_clean)
                output = model.predict(feature_future)
            
            elif model_name == "xgboost_gridsearch":
        
                # Uncomment the grid search code to use GridSearchCV for hyperparameter tuning


                # Define the parameter grid
                param_grid = {
                    'n_estimators': [ 50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.00,0.01, 0.1, 0.2],
                    'lambda': [0.0, 0.1, 1.0],
                    'alpha': [0.0, 0.1, 1.0],
                    'subsample': [0.5, 0.75, 1.0],

                }
                tss = TimeSeriesSplit(n_splits=3, max_train_size=None, test_size=None, gap=0)


                
                # Initialize the estimator
                xgb_estimator = XGBRegressor(objective='reg:squarederror', random_state=42)

                ps = PredefinedSplit(test_fold = consumption_clean)
                
                # Set up the grid search with 3-fold CV
                grid_search = GridSearchCV(
                    estimator=xgb_estimator,
                    param_grid=param_grid,
                    cv=tss,
                    scoring='neg_mean_absolute_error',
                    verbose=1
                )
                
                # Perform grid search
                grid_search.fit(feature_past, consumption_clean)
                
                # Output the best parameters
                print("Best parameters found: ", grid_search.best_params_)
                
                # Retrieve the best estimator and predict
                best_model = grid_search.best_estimator_
                output = best_model.predict(feature_future)

            elif model_name == "multi_time_series":
                # Multi Time Series Model
                model = MultiTimeSeriesForecaster()
                model.train(consumption_clean, feature_past)
                output = model.predict(feature_future)

            else:
            
                print("NO Model specified sofar")
                raise ValueError("No model specified")
            
   
        
            forecast[costumer] = output

    """
    END OF THE MODIFIABLE PART.
    """

    # test to make sure that the output has the expected shape.
    dummy_error = np.abs(forecast - test_set)

    numpy_error = dummy_error.to_numpy()
    customer_id = dummy_error.columns.tolist()

    # 1. Mean error for each customer (mean along the time axis)
    mean_error_per_customer = np.mean(numpy_error, axis=0)

    # 2. Max of the mean errors for each customer
    max_mean_error = np.max(mean_error_per_customer)
    max_mean_error_index = np.argmax(mean_error_per_customer)

    # 3. Overall mean error for all customers (mean across both time and customers)
    overall_mean_error = np.mean(numpy_error)

    # 4. Single maximum error from all the points
    max_error = np.max(numpy_error)

    print("\n")
    print(f"Total Error: {dummy_error.sum().sum()}")
    print(f"Max Mean Error: {max_mean_error:.4f} by: {customer_id[max_mean_error_index]}")
    print(f"Overall Mean Error:{overall_mean_error:.4f}")
    print(f"Max Error:{max_error:.4f}")

    assert np.all(forecast.columns == test_set.columns), (
        "Wrong header or header order."
    )
    assert np.all(forecast.index == test_set.index), (
        "Wrong index or index order."
    )

    assert forecast.isna().sum().sum() == 0, "NaN in forecast."
    
    # Your solution will be evaluated using
    # forecast_error = np.abs(forecast - testing_set).sum().sum(),
    # and then doing a weighted sum the two portfolios:
    # score = forecast_error_IT + 5 * forecast_error_ES

    #make sure the output path exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    logging_folder = "logs"
    if not os.path.exists(logging_folder):
        os.makedirs(logging_folder)
    

    #log results with the name of the model that was used, the country and the error as a csv file aswell as the testset datetime start 
    #and end time date

    log_results(model_name=model_name,
                country=zone,
                error=dummy_error,
                test_start=range_forecast[0],
                test_end = range_forecast[-1],
                logging_folder=logging_folder,
                feature_sets = feature_sets,
                global_model = global_model,
                encoding = encoding_name,
                )



    forecast.to_csv(
        join(output_path, "students_results_" + team_name + "_" + country + ".csv")
    )

if __name__ == "__main__":
    country = "IT"  # it can be ES or IT
    split_date = "2024-08-01 00:00:00"
    train_test = True
    features = ["spv", "temp", "holiday"]  # choice = ["temp", "holiday"]
    global_model = False
    inputation_method = "mean"
    start_date = None
    single_run = True


    if single_run == True:
        print("------------------------------------------------------")
        print("Running the models for the country: ", country)
        print("------------------------------------------------------")
        print("Using the imputation method: ", inputation_method)
        print("Using the start date: ", start_date)
        main(country,
            encoding_name="calculate_custom_features",
            model_name="ligthgbm",
            train_test=train_test,
            split_date=split_date,
            feature_sets=features,
            global_model=global_model,
            inputation_method=inputation_method,
            start_date=start_date
            )
