import pandas as pd
import statsmodels.api as sm
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

class RedemptionModel:

    def __init__(self, X, target_col, scaler, floor=0.0, cap=1.0):
        '''
        Args:
        X (pandas.DataFrame): Dataset of predictors, output from load_data()
        target_col (str): column name for target variable
        '''
        self._predictions = {}
        self.X = X
        self.target_col = target_col
        self.scaler = scaler # store the fitted scaler
        self.floor = floor
        self.cap = cap
        self.results = {} # dict of dicts with model results

    def score(self, truth, preds):
        return MAPE(truth.values.reshape(-1, 1), preds.values.reshape(-1, 1))


    def run_models(self, n_splits=4, test_size=365):
        '''run the models and store results for cross validated splits in
        self.results.
        '''
        #time series split
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        cnt = 0 # keep track of splits
        for train, test in tscv.split(self.X):
            X_train = self.X.iloc[train]
            X_test = self.X.iloc[test]
            #base model
            preds_scaled_base = self._base_model(X_train, X_test)
            #inverse transform predictions for plotting
            preds_original_base = pd.Series(
                self.scaler.inverse_transform(preds_scaled_base.values.reshape(-1, 1)).flatten(),
                index=preds_scaled_base.index
            )
            if 'Base' not in self.results:
                self.results['Base'] = {}
            #score
            self.results['Base'][cnt] = self.score(X_test[self.target_col], preds_scaled_base)
            self.plot(preds_original_base, 'Base', split_num=cnt)

            #prophet model
            preds_scaled_prophet = self._prophet_model(X_train, X_test)
            #inverse transform predictions for plotting
            preds_original_prophet = pd.Series(
                self.scaler.inverse_transform(preds_scaled_prophet.values.reshape(-1, 1)).flatten(),
                index=preds_scaled_prophet.index
            )
            if 'Prophet' not in self.results:
                self.results['Prophet'] = {}
            #score
            self.results['Prophet'][cnt] = self.score(X_test[self.target_col], preds_scaled_prophet)
            self.plot(preds_original_prophet, 'Prophet', color='blue', split_num=cnt)
            cnt += 1

    def _base_model(self, train, test):
        '''
        Our base, too-simple model.
        Your model needs to take the training and test datasets (dataframes)
        and output a prediction based on the test data.

        Please leave this method as-is.

        '''
        res = sm.tsa.seasonal_decompose(train[self.target_col], period=365)
        #copy the seasonal part of the dataset
        seasonal = res.seasonal.copy()
        #center the seasonal component (to avoid shift)
        seasonal = seasonal - seasonal.mean()
        #map dayofyear to seasonal value
        seasonal.index = seasonal.index.dayofyear
        seasonal_avg = seasonal.groupby(seasonal.index).mean()
        seasonal_dict = seasonal_avg.to_dict()
        #estimate baseline (mean) level of target in training
        base_level = train[self.target_col].mean()
        #final prediction = base level + seasonal
        return pd.Series(index=test.index,
                         data=[base_level + seasonal_dict[doy] for doy in test.index.dayofyear])

    def _prophet_model(self, train, test):
        # reset the index so 'Timestamp' becomes a regular column,
        # then rename 'Timestamp' to 'ds' (required by Prophet) and the target column to 'y'
        df_train = train.reset_index().rename(columns={'Timestamp': 'ds', self.target_col: 'y'})
        # do the same for the test set, but we only need the 'ds' column for forecasting
        df_test = test.reset_index().rename(columns={'Timestamp': 'ds'})

        model = Prophet(
            growth='logistic',
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=15.0,
            seasonality_mode='additive'
        )
        #Prophet requires 'cap' and 'floor' columns in the DataFrame for logistic growth.
        df_train['cap'] = self.cap
        df_train['floor'] = self.floor
        df_test['cap'] = self.cap
        df_test['floor'] = self.floor
        #fit the Model
        model.fit(df_train)
        #prepare Future DataFrame for Prediction
        future = df_test[['ds', 'cap', 'floor']]
        #make forecasts
        forecast = model.predict(future)
        #extract predictions and post-process
        preds = forecast['yhat'].values
        return pd.Series(preds, index=test.index).clip(lower=self.floor, upper=self.cap)

    def plot(self, preds_original, label, color='red', split_num=None):
        # plot out the forecasts
        fig, ax = plt.subplots(figsize=(15, 5))
        #inverse transform the FULL original series for plotting
        full_original_data_values = self.scaler.inverse_transform(self.X[self.target_col].values.reshape(-1, 1)).flatten()
        full_original_data_series = pd.Series(full_original_data_values, index=self.X.index)
        ax.scatter(full_original_data_series.index, full_original_data_series.values, s=0.4, color='grey',
            label='Observed')
        #plot predictions (now inverse transformed)
        ax.plot(preds_original.index, preds_original.values, label = label + ' (Forecast)', color=color)

        #add a vertical line to indicate the split point (start of the test set)
        if not preds_original.empty:
            split_point = preds_original.index.min()
            ax.axvline(split_point, color='green', linestyle='--', linewidth=1.5, label='Split Point')

        title_suffix = f" (Split {split_num})" if split_num is not None else ""
        ax.set_title(f'Forecasts for {label} Model{title_suffix}')
        ax.set_ylabel(f'{self.target_col}')
        ax.set_xlabel('Timestamp')
        plt.legend()
        plt.grid(True)
        plt.show()
