import pandas as pd
import numpy as np
from os.path import join
import os
import csv

# depending on your IDE, you might need to add datathon_eth. in front of data
from data import DataLoader, SimpleEncoding, log_results, ImputationEncoding, FeatureEncoding
# depending on your IDE, you might need to add datathon_eth. in front of forecast_models
from forecast_models import SimpleModel, ARIMAModel, GaussianProcessModel, XGBoostModel

def main(zone: str, encoding_name: str, model_name: str, train_test: bool, split_date :str , feature_sets:str, global_model: bool, inputation_method:str = "ffill"):
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
    end_training = training_set.index.max()
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

                pass
        
            elif encoding_name=="calculate_custom_features":

                encoding = FeatureEncoding(
                    consumption, feature_dummy, start_training, end_training, start_forecast, end_forecast, customer=costumer
                )

                feature_past, feature_future, consumption_clean, consumption_test = encoding.calculate_features()

                print(feature_past, feature_future)

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

            elif model_name == "gaussian_process":
                # Gaussian Process Model
                model = GaussianProcessModel()
                model.train(feature_past, consumption_clean)
                output = model.predict(feature_future)

            elif model_name == "xgboost":

                model = XGBoostModel()
                model.train(feature_past, consumption_clean)
                output = model.predict(feature_future)

            else:
            
                print("NO Model specified sofar")
                raise ValueError("No model specified")
            
   
        
            forecast[costumer] = output

    """
    END OF THE MODIFIABLE PART.
    """

    # test to make sure that the output has the expected shape.
    dummy_error = np.abs(forecast - test_set).sum().sum()
    assert np.all(forecast.columns == test_set.columns), (
        "Wrong header or header order."
    )
    assert np.all(forecast.index == test_set.index), (
        "Wrong index or index order."
    )

    assert isinstance(dummy_error, np.float64), "Wrong dummy_error type."
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

    print(f'Dummy Error {dummy_error}')

if __name__ == "__main__":
    country = "ES"  # it can be ES or IT
    split_date = "2024-07-01 00:00:00"
    train_test = True
    features = ["temp"]
    global_model = False
    inputation_method = "mean"

    main(country,
        encoding_name="calculate_custom_features",
        model_name="xgboost",
        train_test=train_test,
        split_date=split_date,
        feature_sets=features,
        global_model=global_model,
        inputation_method=inputation_method
        )
