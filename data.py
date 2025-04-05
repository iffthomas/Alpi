import pandas as pd
from os.path import join

from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self, path: str):
        self.path = path

    def load_data(self, country: str):
        date_format = "%Y-%m-%d %H:%M:%S"

        consumptions_path = join(self.path, "historical_metering_data_" + country + ".csv")

        features_path = join(self.path, "spv_ec00_forecasts_es_it.xlsx")
        example_solution_path = join(self.path, "example_set_" + country + ".csv")

        rollout_path = join(self.path, "rollout_data_" + country + ".csv")

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
        
        #drop all indices with NanValue in the consumption and then drop the same indices in the features



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

        consumptions_path = join(self.path, "historical_metering_data_" + country + ".csv")

        features_path = join(self.path, "spv_ec00_forecasts_es_it.xlsx")
        example_solution_path = join(self.path, "example_set_" + country + ".csv")

        rollout_path = join(self.path, "rollout_data_" + country + ".csv")
        rollout = pd.read_csv(rollout_path, index_col=0, parse_dates=True, date_format=date_format)
        rollout = rollout.reset_index()

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



class SimpleEncoding:
    """
    This class is an example of dataset encoding.

    """

    def __init__(
        self,
        consumption: pd.Series,
        features: pd.Series,
        end_training,
        start_forecast,
        end_forecast,
    ):
        self.consumption_mask = ~consumption.isna()
        self.consumption = consumption[self.consumption_mask]
        self.features = features
        self.end_training = end_training
        self.start_forecast = start_forecast
        self.end_forecast = end_forecast

    def meta_encoding(self):
        """
        This function returns the feature, split between past (for training) and future (for forecasting)),
        as well as the consumption, without missing values.
        :return: three numpy arrays

        """
        features_past = self.features[: self.end_training].values.reshape(-1, 1)
        features_future = self.features[
            self.start_forecast : self.end_forecast
        ].values.reshape(-1, 1)

        features_past = features_past[self.consumption_mask]
        return features_past, features_future, self.consumption


class ImputationEncoding:
    def __init__(self, consumption: pd.DataFrame, features: pd.DataFrame, start_training, end_training, start_forecast, end_forecast, method):
        self.consumption = consumption
        self.features = features
        self.start_training = start_training
        self.end_training = end_training
        self.start_forecast = start_forecast
        self.end_forecast = end_forecast
        self.consumption_mask = ~consumption.isna()
        self.method = method

    def meta_encoding(self):


        if self.method == KNNImputer:
            # Impute missing values in the features using KNNImputer
            imputer = KNNImputer(n_neighbors=5, weights="uniform")
            self.features = pd.DataFrame(imputer.fit_transform(self.features), columns=self.features.columns, index=self.features.index)
            self.consumption = pd.Series(imputer.fit_transform(self.consumption.values.reshape(-1, 1)).flatten(), index=self.consumption.index)
        else:
            for feature in self.features.columns:

                if self.method == "ffill_bbfill":
                    # Impute missing values in the feature column using forward fill
                    self.features[feature] = self.features[feature].ffill()
                    self.consumption = self.consumption.ffill()

                    self.features[feature] = self.features[feature].bfill()
                    self.consumption = self.consumption.bfill()

                    if self.consumption.isna().any():
                        #imput remaining missing values with mean
                        self.consumption = self.consumption.fillna(self.consumption.mean())
                        self.features[feature] = self.features[feature].fillna(self.features[feature].mean())
                    
                elif self.method == "bfill":
                    # Impute missing values in the feature column using backward fill
                    self.features[feature] = self.features[feature].bfill()
                    self.consumption = self.consumption.bfill()
                elif self.method == "mean":
                    # Impute missing values in the feature column using mean
                    self.features[feature] = self.features[feature].fillna(self.features[feature].mean())
                    self.consumption = self.consumption.fillna(self.consumption.mean())


                    

            
        #get the features past and future
        features_past = self.features[self.start_training : self.end_training]

        features_future = self.features[self.start_forecast : self.end_forecast]

        consumption_past = self.consumption[self.start_training : self.end_training]
        consumption_future = self.consumption[self.start_forecast : self.end_forecast]

        return features_past, features_future, consumption_past, consumption_future


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
        self.features = self.features.drop(columns=["DATETIME_COPY"])


        rollout = np.abs(self.features[customer_name] - self.consumption)
        rollout = rollout[rollout > 0].sum()

        if rollout > 200:
            #drop customer_name column
            self.features = self.features.drop(columns=[customer_name])
        



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




        

import os
import csv
from datetime import datetime

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