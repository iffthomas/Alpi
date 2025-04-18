import pandas as pd
import numpy as np
from os.path import join
import os
# depending on your IDE, you might need to add datathon_eth. in front of data
from data import DataLoader, log_results, FeatureEncoding
# depending on your IDE, you might need to add datathon_eth. in front of forecast_models
from forecast_models import LightGBMModel
import warnings
warnings.filterwarnings("ignore")
def main(zone: str, encoding_name: str, model_name: str, train_test: bool, split_date :str , feature_sets:str, start_date:str = None):
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
        training_set, test_set, train_features, test_features = loader.load_data_train_test(country = country, split_date=split_date)
        
        #join train_features and test_features
        features_train_and_test = pd.concat([train_features,test_features])

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


        if encoding_name=="calculate_custom_features":

            encoding = FeatureEncoding(
                consumption, feature_dummy, start_training, end_training, start_forecast, end_forecast, customer=costumer
            )

            feature_past, feature_future, consumption_clean, consumption_test = encoding.calculate_features()

        else:
            print("NO Encoding specified sofar")
            raise ValueError("No encoding specified")

        if model_name == "ligthgbm":
            # LightGBM Model
            model = LightGBMModel()
            model.train(feature_past, consumption_clean)
            output = model.predict(feature_future)
        else:
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
                encoding = encoding_name,
                )



    forecast.to_csv(join(output_path, "students_results_" + team_name + "_" + country + ".csv"))

if __name__ == "__main__":
    country = "IT"  # it can be ES or IT
    split_date = "2024-08-01 00:00:00"
    train_test = True
    features = ["temp"]
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
            start_date=start_date
            )
