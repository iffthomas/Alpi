import pandas as pd
from os.path import join

from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np

def impute_with_past_year_window(df, feature_cols, window_days=7):
    """
    Impute missing values in multiple feature columns using a past-year +/- window strategy.
    
    Parameters:
    - df: DataFrame with datetime index
    - feature_cols: list of column names to impute
    - window_days: number of days before and after the past-year timestamp
    
    Returns:
    - DataFrame with imputed values
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    for feature in feature_cols:
        series = df[feature].copy()
        missing_idx = series[series.isna()].index

        for timestamp in missing_idx:
            past_year = timestamp - pd.DateOffset(years=1)
            hour = timestamp.hour

            start = past_year - pd.Timedelta(days=window_days)
            end = past_year + pd.Timedelta(days=window_days)

            # Filter the window with same hour
            mask = (df.index >= start) & (df.index <= end) & (df.index.hour == hour)
            window_values = df.loc[mask, feature].dropna()

            if not window_values.empty:
                series[timestamp] = window_values.mean()

        df[feature] = series

    return df

class DataLoader:
    def __init__(self, path: str):
        self.path = path

    def load_data(self, country: str):
        date_format = "%Y-%m-%d %H:%M:%S"

        consumptions_path = join(self.path, "historical_metering_data_" + country + ".csv")

        features_path = join(self.path, "spv_ec00_forecasts_es_it.xlsx")
        example_solution_path = join(self.path, "example_set_" + country + ".csv")

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
        example_solution = pd.read_csv(
            example_solution_path,
            index_col=0,
            parse_dates=True,
            date_format=date_format,
        )

        return consumptions, features, example_solution

    def load_data_train_test(self, country, split_date):

        date_format = "%Y-%m-%d %H:%M:%S"

        consumptions_path = join(self.path, "historical_metering_data_" + country + ".csv")

        features_path = join(self.path, "spv_ec00_forecasts_es_it.xlsx")
        example_solution_path = join(self.path, "example_set_" + country + ".csv")

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


        # Convert split_date to a datetime object if it's not already one
        split_date = pd.to_datetime(split_date, format=date_format)

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
        features_past = self.features[: self.end_training].values
        features_future = self.features[
            self.start_forecast : self.end_forecast
        ].values

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

                if self.method == "ffill":
                    # Impute missing values in the feature column using forward fill
                    self.features[feature] = self.features[feature].ffill()
                    self.consumption = self.consumption.ffill()
                elif self.method == "bfill":
                    # Impute missing values in the feature column using backward fill
                    self.features[feature] = self.features[feature].bfill()
                    self.consumption = self.consumption.bfill()
                elif self.method == "mean":
                    # Impute missing values in the feature column using mean
                    self.features[feature] = self.features[feature].fillna(self.features[feature].mean())
                    self.consumption = self.consumption.fillna(self.consumption.mean())
                elif self.method == "past-year":
                    # Impute missing values in the feature column using past-year +/- window strategy
                    self.features = impute_with_past_year_window(self.features, [feature])
                    self.consumption = impute_with_past_year_window(self.consumption.to_frame(), ["consumption"])


                    

            
        #get the features past and future
        features_past = self.features[self.start_training : self.end_training]

        features_future = self.features[self.start_forecast : self.end_forecast]

        consumption_past = self.consumption[self.start_training : self.end_training]
        consumption_future = self.consumption[self.start_forecast : self.end_forecast]

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