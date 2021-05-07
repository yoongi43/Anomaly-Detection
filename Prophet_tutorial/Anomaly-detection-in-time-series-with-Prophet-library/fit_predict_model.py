import matplotlib.pyplot as plt
from prophet import Prophet


def fit_predict_model(dataframe, interval_width=0.99, changepoint_range=0.8):
    m = Prophet(daily_seasonality=False, yearly_seasonality=False, weekly_seasonality=False,
                seasonality_mode='multiplicative',
                interval_width=interval_width,
                changepoint_range=changepoint_range)
    m = m.fit(dataframe)
    forecast = m.predict(dataframe)
    forecast['fact'] = dataframe['y'].reset_index(drop=True)

    return forecast