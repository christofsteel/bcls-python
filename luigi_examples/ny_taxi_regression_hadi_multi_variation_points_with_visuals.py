import os
import pickle
import warnings
from pathlib import Path
import luigi
import numpy as np
import pandas as pd
import json
from pickle import dump

from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler, MinMaxScaler
from sklearn.exceptions import ConvergenceWarning
from inhabitation_task import LuigiCombinator, ClsParameter, RepoMeta
from cls_python import FiniteCombinatoryLogic, Subtypes
from cls_luigi_read_tabular_data import WriteSetupJson, ReadTabularData

import seaborn as sns
import matplotlib.pyplot as plt

from unique_task_pipeline_validator import UniqueTaskPipelineValidator

sns.set_style('darkgrid')
sns.set_context('talk')

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)


def read_setup(path='data/setup.json'):
    with open(path, 'rb') as f:
        setup = json.load(f)
    return setup


class WriteCSVRegressionSetupJson(WriteSetupJson):
    abstract = False

    def run(self):
        d = {
            "csv_file": "data/taxy_trips_ny_2016-06-01to03_3%sample.csv",
            "date_column": ['pickup_datetime', 'dropoff_datetime'],
            # "temperature_column": ["max_temp", "min_temp"],
            "drop_column": ["rain", "has_rain", "main_street",
                            "main_street_ratio", "trip_distance",
                            'pickup_datetime', 'dropoff_datetime',
                            'vendor_id', "passenger_count", "max_temp", "min_temp"],
            "target_column": 'trip_duration',
            "seed": 42
        }
        with open(self.output().path, 'w') as f:
            json.dump(d, f, indent=4)


class ReadTaxiData(ReadTabularData):
    abstract = False

    def run(self):
        setup = self._read_setup()
        taxi = pd.read_csv(setup["csv_file"], parse_dates=setup["date_column"])
        taxi.to_pickle(self.output().path)


class InitialCleaning(luigi.Task, LuigiCombinator):
    abstract = False
    tabular_data = ClsParameter(tpe=ReadTaxiData.return_type())

    def requires(self):
        return [self.tabular_data()]

    def run(self):
        taxi = pd.read_pickle(self.input()[0].open().name)
        print("Taxi DataFrame shape before dropping NaNs and duplicates", taxi.shape)
        taxi = taxi.dropna().drop_duplicates()
        print("Taxi DataFrame shape after dropping NaNs and duplicates", taxi.shape)
        taxi.to_pickle(self.output().path)

    def output(self):
        return luigi.LocalTarget('data/cleaned_data.pkl')


class FilterImplausibleTrips(luigi.Task, LuigiCombinator):
    abstract = False
    clean_data = ClsParameter(tpe=InitialCleaning.return_type())

    def requires(self):
        return [self.clean_data()]

    def run(self):
        setup = read_setup()
        taxi = pd.read_pickle(self.input()[0].open().name)
        taxi = taxi[
            (taxi[setup["target_column"]] >= 10) &
            (taxi[setup["target_column"]] <= 100000)
            ]
        print("Taxi DataFrame shape after filtering implausible trips", taxi.shape)
        taxi.to_pickle(self.output().path)

    def output(self):
        return luigi.LocalTarget('data/filtered.pkl')


class ExtractRawTemporalFeatures(luigi.Task, LuigiCombinator):
    abstract = False
    filtered_tabular_data = ClsParameter(tpe=FilterImplausibleTrips.return_type())

    def requires(self):
        return [self.filtered_tabular_data()]

    def _read_tabular_data(self):
        return pd.read_pickle(self.input()[0].open().name)

    def run(self):
        setup = read_setup()
        tabular = self._read_tabular_data()
        raw_temporal_features = pd.DataFrame(index=tabular.index)
        for c in setup["date_column"]:
            print("Preprocessing Datetime-Column:", c)
            raw_temporal_features[c + "_YEAR"] = tabular[c].dt.year
            raw_temporal_features[c + "_MONTH"] = tabular[c].dt.hour
            raw_temporal_features[c + "_DAY"] = tabular[c].dt.day
            raw_temporal_features[c + "_WEEKDAY"] = tabular[c].dt.dayofweek
            raw_temporal_features[c + "_HOUR"] = tabular[c].dt.hour

        raw_temporal_features.to_pickle(self.output().path)

    def output(self):
        return luigi.LocalTarget('data/raw_temporal_features.pkl')


class BinaryEncodePickupAtWeekend(luigi.Task, LuigiCombinator):
    abstract = False
    raw_temporal_data = ClsParameter(tpe=ExtractRawTemporalFeatures.return_type())
    weekend_start_ix = luigi.IntParameter(default=5)

    def requires(self):
        return [self.raw_temporal_data()]

    def _read_tabular_data(self):
        return pd.read_pickle(self.input()[0].open().name)

    def run(self):
        raw_temporal_data = self._read_tabular_data()
        df_is_weekend = pd.DataFrame(index=raw_temporal_data.index)

        def weekend_mapping(weekday):
            if weekday >= self.weekend_start_ix:
                return 1
            return 0

        df_is_weekend["pickup_at_weekend"] = raw_temporal_data["pickup_datetime_WEEKDAY"].map(
            weekend_mapping)
        df_is_weekend.to_pickle(self.output().path)
        print('NOW WE BINARY ENCODE Weekend')

    def output(self):
        return luigi.LocalTarget('data/pickup_at_weekend.pkl')


class BinaryEncodePickupAtHour(luigi.Task, LuigiCombinator):
    abstract = False
    raw_temporal_data = ClsParameter(tpe=ExtractRawTemporalFeatures.return_type())
    hour = luigi.IntParameter(default=7)

    def requires(self):
        return [self.raw_temporal_data()]

    def _read_tabular_data(self):
        return pd.read_pickle(self.input()[0].open().name)

    def run(self):
        raw_temporal_data = self._read_tabular_data()
        df_pickup_hour_encoded = pd.DataFrame(index=raw_temporal_data.index)

        def pickup_hour_mapper(hour):
            if hour >= self.hour:
                return 1
            return 0

        col_name = "is_or_after_" + str(self.hour)
        df_pickup_hour_encoded[col_name] = raw_temporal_data["pickup_datetime_HOUR"].map(
            pickup_hour_mapper)
        df_pickup_hour_encoded.to_pickle(self.output().path)

    def output(self):
        return luigi.LocalTarget('data/pickup_hour_' + str(self.hour) + '_binary.pkl')


class EncodePickupWeekdayOneHotSklearn(luigi.Task, LuigiCombinator):
    abstract = False
    raw_temporal_data = ClsParameter(tpe=ExtractRawTemporalFeatures.return_type())

    def requires(self):
        return [self.raw_temporal_data()]

    def _read_tabular_data(self):
        return pd.read_pickle(self.input()[0].open().name)

    def run(self):
        raw_temporal_data = self._read_tabular_data()

        def map_weekday_num_to_name(weekday_num):
            weekdays = {
                0: "Monday",
                1: "Tuesday",
                2: "Wednesday",
                3: "Thursday",
                4: "Friday",
                5: "Saturday",
                6: "Sunday"
            }
            return "pickup " + weekdays[weekday_num]

        raw_temporal_data["pickup_datetime_WEEKDAY"] = raw_temporal_data["pickup_datetime_WEEKDAY"].map(
            map_weekday_num_to_name
        )
        transformer = OneHotEncoder(sparse=False)
        encoded_features = transformer.fit_transform(raw_temporal_data[["pickup_datetime_WEEKDAY"]])

        category_columns = np.concatenate(transformer.categories_)
        onehot_weekdays = pd.DataFrame(
            encoded_features,
            columns=category_columns,
            index=raw_temporal_data.index)

        onehot_weekdays.to_pickle(self.output().path)
        print('NOW WE ONE HOT ENCODE WEEKDAY')

    def output(self):
        return luigi.LocalTarget('data/one_hot_weekday.pkl')


# class DummyEncodingNode(BinaryEncodePickupIsInWeekend,
#                         EncodePickupWeekdayOneHotSklearn):
#     abstract = False
#
#     def run(self):
#         with open(self.output().path, "w") as f:
#             f.write("Don't Mind Me; I'm Just A Dummy:D")
#
#     def output(self):
#         return luigi.LocalTarget('data/no_encoding')
#
#
# s = {BinaryEncodePickupIsInWeekend,
#      EncodePickupWeekdayOneHotSklearn}

class AssembleFinalDataset(luigi.Task, LuigiCombinator):
    abstract = True
    filtered_trips = ClsParameter(tpe=FilterImplausibleTrips.return_type())
    raw_temporal_features = ClsParameter(tpe=ExtractRawTemporalFeatures.return_type())
    is_after_7am = ClsParameter(tpe=BinaryEncodePickupAtHour.return_type())

    def _get_variant_label(self):
        var_label_name = list(filter(
            lambda local_target: "no_encoding" not in local_target.path, self.input()))
        var_label_name = list(map(
            lambda local_target: Path(local_target.path).stem, var_label_name))
        return "_".join(var_label_name)

    def run(self):
        setup = read_setup()
        df_joined = None
        for i in self.input():
            path = i.open().name
            # if 'no_encoding' not in path:
            if df_joined is None:
                df_joined = pd.read_pickle(path)
            else:
                data = pd.read_pickle(path)
                df_joined = pd.merge(df_joined, data, left_index=True, right_index=True)

        if len(setup["drop_column"]) != 0:
            df_joined = df_joined.drop(setup["drop_column"], axis="columns")
        df_joined.to_pickle(self.output().path)

    def output(self):
        return luigi.LocalTarget("data/final_" + self._get_variant_label() + ".pkl")


class AssembleFinalDataset1(AssembleFinalDataset):
    abstract = False
    is_weekend = ClsParameter(tpe=BinaryEncodePickupAtWeekend.return_type())

    def requires(self):
        return [self.filtered_trips(), self.raw_temporal_features(),
                self.is_after_7am(), self.is_weekend()]


class AssembleFinalDataset2(AssembleFinalDataset):
    abstract = False
    onehot_weekdays = ClsParameter(tpe=EncodePickupWeekdayOneHotSklearn.return_type())

    def requires(self):
        return [self.filtered_trips(), self.raw_temporal_features(),
                self.is_after_7am(), self.onehot_weekdays()]


class AssembleFinalDataset3(AssembleFinalDataset):
    abstract = False

    def requires(self):
        return [self.filtered_trips(), self.raw_temporal_features(),
                self.is_after_7am()]


class AssembleFinalDataset4(AssembleFinalDataset):
    abstract = False
    is_weekend = ClsParameter(tpe=BinaryEncodePickupAtWeekend.return_type())
    onehot_weekdays = ClsParameter(tpe=EncodePickupWeekdayOneHotSklearn.return_type())

    def requires(self):
        return [self.filtered_trips(), self.raw_temporal_features(),
                self.is_after_7am(), self.onehot_weekdays(), self.is_weekend()]


class TrainTestSplit(luigi.Task, LuigiCombinator):
    abstract = False
    final_dataset = ClsParameter(tpe=AssembleFinalDataset.return_type())

    def requires(self):
        return [self.final_dataset()]

    def _get_variant_label(self):
        return Path(self.input()[0].path).stem

    def _read_tabular(self):
        return pd.read_pickle(self.input()[0].open().name)

    def run(self):
        tabular = self._read_tabular()
        setup = read_setup()

        train, test = train_test_split(tabular, test_size=0.33, random_state=setup['seed'])

        train.to_pickle(self.output()[0].path)
        test.to_pickle(self.output()[1].path)

    def output(self):
        return [
            luigi.LocalTarget('data/train_' + self._get_variant_label() + '.pkl'),
            luigi.LocalTarget('data/test_' + self._get_variant_label() + '.pkl')
        ]


class FitTransformScaler(luigi.Task, LuigiCombinator):
    abstract = True
    splitted_data = ClsParameter(tpe=TrainTestSplit.return_type())

    def requires(self):
        return [self.splitted_data()]

    def _read_training_data(self):
        return pd.read_pickle(self.input()[0][0].open().name)

    def _read_testing_data(self):
        return pd.read_pickle(self.input()[0][1].open().name)

    def _get_training_variant_label(self):
        return Path(self.input()[0][0].path).stem

    def _get_testing_variant_label(self):
        return Path(self.input()[0][1].path).stem


class FitTransformRobustScaler(FitTransformScaler):
    abstract = False

    def run(self):
        scaler = RobustScaler()
        setup = read_setup()

        train = self._read_training_data()
        X = train.drop(setup["target_column"], axis="columns")
        y = train[[setup["target_column"]]]
        scaler.fit(X)
        scaled = pd.DataFrame(scaler.transform(X),
                              columns=scaler.feature_names_in_,
                              index=X.index)
        scaled[setup["target_column"]] = y
        scaled.to_pickle(self.output()[0].path)

        test = self._read_testing_data()
        X = test.drop(setup["target_column"], axis="columns")
        y = test[[setup["target_column"]]]
        scaled = pd.DataFrame(scaler.transform(X),
                              columns=scaler.feature_names_in_,
                              index=X.index)
        scaled[setup["target_column"]] = y
        scaled.to_pickle(self.output()[1].path)

        with open(self.output()[2].path, 'wb') as outfile:
            pickle.dump(scaler, outfile)

    def output(self):
        return [
            # scaled training data
            luigi.LocalTarget('data/' + self._get_training_variant_label() + '_robust_scaled' + '.pkl'),
            # scaled testing data
            luigi.LocalTarget('data/' + self._get_testing_variant_label() + '_robust_scaled' + '.pkl'),
            # scaler it self
            luigi.LocalTarget('data/scaler_robust_' + self._get_training_variant_label() + '.pkl')
        ]


class FitTransformMinMaxScaler(FitTransformScaler):
    abstract = False

    def run(self):
        scaler = MinMaxScaler()
        setup = read_setup()

        train = self._read_training_data()
        X = train.drop(setup["target_column"], axis="columns")
        y = train[[setup["target_column"]]]
        scaler.fit(X)
        scaled = pd.DataFrame(scaler.transform(X),
                              columns=scaler.feature_names_in_,
                              index=X.index)
        scaled[setup["target_column"]] = y
        scaled.to_pickle(self.output()[0].path)

        test = self._read_testing_data()
        X = test.drop(setup["target_column"], axis="columns")
        y = test[[setup["target_column"]]]
        scaled = pd.DataFrame(scaler.transform(X),
                              columns=scaler.feature_names_in_,
                              index=X.index)
        scaled[setup["target_column"]] = y
        scaled.to_pickle(self.output()[1].path)

        with open(self.output()[2].path, 'wb') as outfile:
            pickle.dump(scaler, outfile)

    def output(self):
        return [
            # scaled training data
            luigi.LocalTarget('data/' + self._get_training_variant_label() + '_minmax_scaled' + '.pkl'),
            # scaled testing data
            luigi.LocalTarget('data/' + self._get_testing_variant_label() + '_minmax_scaled' + '.pkl'),
            # scaler it self
            luigi.LocalTarget('data/scaler_minmax_' + self._get_training_variant_label() + '.pkl')
        ]


class TrainRegressionModel(luigi.Task, LuigiCombinator):
    abstract = True
    scaled_data = ClsParameter(tpe=FitTransformScaler.return_type())

    def requires(self):
        return [self.scaled_data()]

    def _get_variant_label(self):
        return Path(self.input()[0][0].path).stem


class TrainLinearRegressionModel(TrainRegressionModel):
    abstract = False

    def run(self):
        setup = read_setup()
        tabular = pd.read_pickle(self.input()[0][0].open().name)
        print("TARGET:", setup["target_column"])
        print("NOW WE FIT LINEAR REGRESSION MODEL")

        X = tabular.drop(setup["target_column"], axis="columns")
        y = tabular[[setup["target_column"]]].values.ravel()
        print("WITH THE FEATURES")
        print(X.columns)
        print(X)
        print(X.shape)
        print("AND Target")
        print(y)
        print(y.shape)
        reg = LinearRegression().fit(X, y)
        with open(self.output()[0].path, 'wb') as f:
            dump(reg, f)

    def output(self):
        return [luigi.LocalTarget('data/reg_linear_' + self._get_variant_label() + '.pkl')]


class TrainLassoRegressionModel(TrainRegressionModel):
    abstract = False

    def run(self):
        setup = read_setup()
        tabular = pd.read_pickle(self.input()[0][0].open().name)
        print("TARGET:", setup["target_column"])
        print("NOW WE FIT LASSO MODEL")

        X = tabular.drop(setup["target_column"], axis="columns")
        y = tabular[[setup["target_column"]]].values.ravel()
        print("WITH THE FEATURES")
        print(X.columns)
        print(X)
        print(X.shape)
        print("AND Target")
        print(y)
        print(y.shape)
        reg = linear_model.Lasso(alpha=0.1, random_state=setup["seed"]).fit(X, y)
        with open(self.output()[0].path, 'wb') as f:
            dump(reg, f)

    def output(self):
        return [luigi.LocalTarget('data/reg_lasso_' + self._get_variant_label() + '.pkl')]


class TrainRidgeRegressionModel(TrainRegressionModel):
    abstract = False

    def run(self):
        setup = read_setup()
        tabular = pd.read_pickle(self.input()[0][0].open().name)
        print("TARGET:", setup["target_column"])
        print("NOW WE FIT LASSO MODEL")

        X = tabular.drop(setup["target_column"], axis="columns")
        y = tabular[[setup["target_column"]]].values.ravel()
        print("WITH THE FEATURES")
        print(X.columns)
        print(X)
        print(X.shape)
        print("AND Target")
        print(y)
        print(y.shape)
        reg = linear_model.Ridge(alpha=0.1, random_state=setup["seed"]).fit(X, y)
        with open(self.output()[0].path, 'wb') as f:
            dump(reg, f)

    def output(self):
        return [luigi.LocalTarget('data/reg_ridge_' + self._get_variant_label() + '.pkl')]


class Predict(luigi.Task, LuigiCombinator):
    abstract = False
    regressor = ClsParameter(tpe=TrainRegressionModel.return_type())
    scaled_data = ClsParameter(tpe=FitTransformScaler.return_type())

    def requires(self):
        return [self.regressor(), self.scaled_data()]

    def _load_regressor(self):
        with open(self.input()[0][0].path, 'rb') as file:
            reg = pickle.load(file)

        return reg

    def _read_scaled_test_data(self):
        p = self.input()[1][1].path
        with open(p, 'rb') as file:
            test_data = pickle.load(file)

        return test_data

    def _get_variant_label(self):
        return Path(self.input()[0][0].path).stem

    def run(self):
        reg = self._load_regressor()
        test_data = self._read_scaled_test_data()
        setup = read_setup()
        y_true_and_pred = pd.DataFrame(index=test_data.index)
        X = test_data.drop(setup["target_column"], axis="columns")
        y_true_and_pred[setup["target_column"]] = test_data[setup["target_column"]].values.ravel()
        y_true_and_pred['prediction'] = reg.predict(X)

        with open(self.output().path, 'wb') as outfile:
            pickle.dump(y_true_and_pred, outfile)

    def output(self):
        return luigi.LocalTarget('data/true_and_prediction_' + self._get_variant_label() + '.pkl')


class EvaluateAndVisualize(luigi.Task, LuigiCombinator):
    abstract = False
    y_true_and_pred = ClsParameter(tpe=Predict.return_type())
    sort_by = luigi.Parameter(default="rmse")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.done = False
        self.rmse = None
        self.mae = None
        self.r2 = None

    def complete(self):
        return self.done

    def requires(self):
        return self.y_true_and_pred()

    def _read_y_true_and_prediction(self):
        setup = read_setup()
        tabular = pd.read_pickle(self.input().path)
        return tabular[setup['target_column']], tabular['prediction']

    def _get_reg_name(self):
        p = Path(self.input().path).name
        p = p.split('_')[3:]
        return '_'.join(p)

    def _update_leaderboard(self):
        if os.path.exists(self.output()[0].path) is False:
            leaderboard = pd.DataFrame(columns=["regressor", "rmse", "mae", "r2"])
        else:
            leaderboard = pd.read_csv(self.output()[0].path, index_col="index")

        if self._get_reg_name() not in leaderboard["regressor"].values:
            leaderboard.loc[leaderboard.shape[0]] = [self._get_reg_name(), self.rmse, self.mae, self.r2]
            leaderboard = leaderboard.sort_values(by=self.sort_by, ascending=False)
            leaderboard.to_csv(self.output()[0].path, index_label="index")

    def _compute_metrics(self, y_true, y_pred):
        self.rmse = round(mean_squared_error(y_true, y_pred, squared=False), 3)
        self.mae = round(mean_absolute_error(y_true, y_pred), 3)
        self.r2 = round(r2_score(y_true, y_pred), 3)

    def _visualize(self, y_true, y_pred, show=False):

        # if os.path.exists(self.output()[1].path) is False:

        fig, axes = plt.subplots(2, 2, figsize=(17, 12))
        fig.suptitle(
            "\nPerformance Evaluation\nRMSE: {}\nMAE: {}\nR\u00b2: {}\n".format(self.rmse, self.mae, self.r2),
            x=0.05, ha="left")

        df_prediction = pd.DataFrame({
            'Actual': y_true,
            'Predicted': y_pred})

        def compute_residual(row):
            return row['Predicted'] - row['Actual']

        df_prediction['Prediction Residual'] = df_prediction.apply(
            compute_residual, axis=1)

        residual_plot = sns.scatterplot(x='Actual', y='Prediction Residual',
                                        data=df_prediction, color='r', ax=axes[0][0])
        residual_plot.set_title('Residual Scatter Plot\nIdeal Situation (Predicted = Actual) in Green', loc='left')
        residual_plot.axhline(
            y=0, color='green',
            ls='-', lw=3)

        scatter_plot = sns.scatterplot(x='Actual', y='Predicted',
                                       data=df_prediction, color='r', ax=axes[0][1])
        scatter_plot.set_title(
            'Predictions -- Ground Truth Scatter plot\n'
            'Ideal Situation (Predicted = Actual) in Green', loc='left')
        xlims = (-10, max(max(y_true), max(y_pred)))
        scatter_plot.plot(xlims, xlims, color='g', ls="-", lw=3)

        histogram_residuals = sns.histplot(data=df_prediction, x='Prediction Residual',
                                           kde=True, stat="density", ax=axes[1][0])
        histogram_residuals.set_title('Residuals Histogram\nIdeal Situation (Predicted = Actual) in Green ',
                                      loc='left')
        histogram_residuals.axvline(
            x=0, color='green',
            ls='-', lw=3)

        df_prediction["i"] = df_prediction.index
        df_histplot = pd.melt(df_prediction, id_vars=['i'],
                              value_vars=['Actual', 'Predicted'],
                              var_name='Curve', value_name='Target Variable')

        histogram_target_value = sns.histplot(data=df_histplot, x='Target Variable', hue="Curve",
                                              kde=True, stat="density", ax=axes[1][1])
        histogram_target_value.set_title('Ground Truth & Prediction Values Histogram', loc='left')

        plt.tight_layout()
        plt.savefig(self.output()[1].path)
        if show:
            plt.show()

    def run(self):
        y_true, y_pred = self._read_y_true_and_prediction()
        self._compute_metrics(y_true, y_pred)
        self._update_leaderboard()
        self._visualize(y_true, y_pred, show=True)
        self.done = True

    def output(self):
        return [luigi.LocalTarget('data/leaderboard.csv'),
                luigi.LocalTarget('data/' + self._get_reg_name() + ".png")]


# class FinalNode(luigi.WrapperTask, LuigiCombinator):
#     evaluate = ClsParameter(tpe=EvaluateAndVisualize.return_type())
#
#     def requires(self):
#         return self.evaluate()


if __name__ == '__main__':

    target = EvaluateAndVisualize.return_type()
    print("Collecting Repo")
    repository = RepoMeta.repository
    print("Build Repository...")
    fcl = FiniteCombinatoryLogic(repository, Subtypes(RepoMeta.subtypes), processes=1)
    print("Build Tree Grammar and inhabit Pipelines...")

    inhabitation_result = fcl.inhabit(target)
    print("Enumerating results...")
    max_tasks_when_infinite = 10
    actual = inhabitation_result.size()
    max_results = max_tasks_when_infinite
    if actual > 0:
        max_results = actual

    validator = UniqueTaskPipelineValidator([AssembleFinalDataset, FitTransformScaler, TrainRegressionModel])
    results = [t() for t in inhabitation_result.evaluated[0:max_results] if validator.validate(t())]

    if results:
        print("Number of results", max_results)
        print("Number of results after filtering", len(results))
        print("Run Pipelines")
        luigi.build(results, local_scheduler=False, detailed_summary=True)
    else:
        print("No results!")
