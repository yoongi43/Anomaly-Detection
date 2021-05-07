#Anomaly detection in time series with Prophet library

## Citation
Codes : https://towardsdatascience.com/anomaly-detection-time-series-4c661f6f165f

Data : https://github.com/facebook/prophet/tree/master/examples

### read_dataframe.py
From example folder, read dataframe.
For example forder of facebook/prophet, data_idx ranges from 0 to 6.
__ds__ is yyyy-mm-dd datetime data and __y__ is values.


### fit_predict_model.py
Using prophet module, create model.
__interval_width__ is the width of the uncertainty intervals provided for the forecast.
__changepoint_range__ is the proportion of history in which trend changepoints will be estimated. 
Using __reset_index(drop=True)__, take only the value and discard the index.

Fit the dataframe to Prophet model and return prediction(forcast).

### detect_anomalies.py
From forcasted dataframe, pick out anomalies.
__importacne__ is a value that indicates how much the __fact__ values differs compared 
to expected (__yhat__) based on __fact__ values. When plotted, it is represented as the size of the red circle.

### plot_anomalies.py
Plot the band(interval), true values (__fact__) and anomalies in them with altair library.

