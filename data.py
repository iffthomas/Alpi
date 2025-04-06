import os
import csv
from datetime import datetime
import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self, path: str):
        self.path = path

    def load_data(self, country: str):
        date_format = "%Y-%m-%d %H:%M:%S"

        consumptions_path = os.path.join(self.path, "historical_metering_data_" + country + ".csv")

        features_path = os.path.join(self.path, "spv_ec00_forecasts_es_it.xlsx")
        example_solution_path = os.path.join(self.path, "example_set_" + country + ".csv")

        rollout_path = os.path.join(self.path, "rollout_data_" + country + ".csv")

        rollout = pd.read_csv(rollout_path, index_col=0, parse_dates=True, date_format=date_format)
    

        consumptions = pd.read_csv(
            consumptions_path, index_col=0, parse_dates=True, date_format=date_format
        )


        features = pd.read_excel(
            features_path,
            sheet_name=country,
            index_col=0,
            parse_dates=True,
            date_format=date_format,
        )

        features = pd.merge(features, rollout, on="DateTime", how="left")

        example_solution = pd.read_csv(
            example_solution_path,
            index_col=0,
            parse_dates=True,
            date_format=date_format,
        )
        example_solution = example_solution.dropna()

        return consumptions, features, example_solution

    def load_data_train_test(self, country, split_date):

        date_format = "%Y-%m-%d %H:%M:%S"

        consumptions_path = os.path.join(self.path, "historical_metering_data_" + country + ".csv")

        if split_date == "2024-08-01 00:00:00":
            example_set_path = f"datasets2025\example_set_{country}.csv"
            example_solution = pd.read_csv(
                example_set_path,
                index_col=0,
                parse_dates=True,
                date_format=date_format,
            )

        features_path = os.path.join(self.path, "spv_ec00_forecasts_es_it.xlsx")

        rollout_path = os.path.join(self.path, "rollout_data_" + country + ".csv")
        rollout = pd.read_csv(rollout_path, index_col=0, parse_dates=True, date_format=date_format)
        rollout = rollout.reset_index()

        consumptions = pd.read_csv(
            consumptions_path, index_col=0, parse_dates=True, date_format=date_format
        )

        if split_date == "2024-08-01 00:00:00":
            print("Splitting here")

            consumptions = pd.concat([consumptions, example_solution])

        features = pd.read_excel(
            features_path,
            sheet_name=country,
            index_col=0,
            parse_dates=True,
            date_format=date_format,
        )

        features = features.reset_index()
        features = features.rename(columns={"index": "DATETIME"})
        features = pd.merge(features, rollout, on="DATETIME", how="left")

        # Convert split_date to a datetime object if it's not already one
        split_date = pd.to_datetime(split_date, format=date_format)

        features["DATETIME"] = pd.to_datetime(features["DATETIME"])
        features["DATETIME_COPY"] = features["DATETIME"]  # optional copy if needed
        features.set_index("DATETIME", inplace=True)


        # Split the data based on the index (date)
        train_consumptions = consumptions[consumptions.index < split_date]
        test_consumptions = consumptions[consumptions.index >= split_date]

        train_features = features[features.index < split_date]
        test_features = features[features.index >= split_date]

        return train_consumptions, test_consumptions, train_features, test_features


class FeatureEncoding:
    def __init__(self, consumption: pd.DataFrame, features: pd.DataFrame, start_training, end_training, start_forecast, end_forecast,customer):
        self.consumption = consumption
        self.features = features
        self.start_training = start_training
        self.end_training = end_training
        self.start_forecast = start_forecast
        self.end_forecast = end_forecast
        self.consumption_mask = ~consumption.isna()
        self.customer = customer


    def calculate_features(self):

        customer_name = self.customer
        customer_name = customer_name.replace("VALUEMWHMETERINGDATA","INITIALROLLOUTVALUE")
        feature_cols = ["spv","temp","holiday","DATETIME_COPY",customer_name]

        self.features = self.features[feature_cols]

        self.features["Month"] = self.features["DATETIME_COPY"].dt.month
        self.features["Day"] = self.features["DATETIME_COPY"].dt.day
        self.features["Hour"] = self.features["DATETIME_COPY"].dt.hour
        self.features["Dayofweek"] = self.features["DATETIME_COPY"].dt.dayofweek
        

        self.features["log_spv"] = np.log(self.features["spv"] + 1e-9)  # Adding a small constant to avoid log(0)
        self.features["log_temp"] = np.log(self.features["temp"] + 1e-9)  # Adding a small constant to avoid log(0)


        self.features["cosine_hour"] = np.cos(2 * np.pi * self.features["Hour"] / 24)
        self.features["sine_hour"] = np.sin(2 * np.pi * self.features["Hour"] / 24)

        self.features["cosine_day"] = np.cos(2 * np.pi * self.features["Day"] / 31)
        self.features["sine_day"] = np.sin(2 * np.pi * self.features["Day"] / 31)

        self.features["cosine_month"] = np.cos(2 * np.pi * self.features["Month"] / 12)
        self.features["sine_month"] = np.sin(2 * np.pi * self.features["Month"] / 12)


        #drop Datetime column
        # Drop the unnecessary column
        self.features = self.features.drop(columns=["DATETIME_COPY"])

        for feature in self.features.columns:
            #if data type is categorical, do mode imputation
            if self.features[feature].dtype == "category":
                continue
            else:
                self.features[feature] = self.features[feature].fillna(self.features[feature].mean())
        self.consumption = self.consumption.fillna(self.consumption.mean())


        # Example feature calculation: mean and standard deviation of consumption
        features_past = self.features[self.start_training : self.end_training]
        features_future = self.features[self.start_forecast : self.end_forecast]

        consumption_past = self.consumption[self.start_training : self.end_training]
        consumption_future = self.consumption[self.start_forecast : self.end_forecast]


        #do imputa

        return features_past, features_future, consumption_past, consumption_future




def log_results(model_name, country, error, test_start, test_end, logging_folder, feature_sets, global_model, encoding):
    log_file = os.path.join(logging_folder,country, f"{model_name}_results.csv")
    #check if folder existswhere i want to safe the file else make it

    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))
        
    file_exists = os.path.isfile(log_file)

    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Write header if file is new
        if not file_exists:
            writer.writerow(["Timestamp", "Model", "Country", "Error", "Test Start", "Test End", "Feature Sets", "Global Model", "Encoding"])
        
        writer.writerow([
            datetime.now().isoformat(),
            model_name,
            country,
            error,
            test_start.isoformat(),
            test_end.isoformat(),
            feature_sets,
            global_model,
            encoding

        ])