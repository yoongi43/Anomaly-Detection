from prophet import Prophet
from read_dataframe import read_dataframe
from forecast import forecast
import matplotlib.pyplot as plt


def main():
    path_example = "../examples"

    """example 1"""
    df = read_dataframe(path_examples=path_example, data_idx=4)
    model, future, forecasted = forecast(df, periods=1096, showflag=True)  # uncertainty intervals seem way too wide

    df.loc[(df['ds'] > '2010-01-01') & (df['ds'] < '2011-01-01'), 'y'] = None
    model_removed = Prophet().fit(df)
    fig = model_removed.plot(model_removed.predict(future))  # model with missing data. prediction of whole data with future.
    fig.set_figheight(18)
    fig.set_figwidth(9)
    plt.title('prediction (model with missing data)')
    plt.show()

    """example 2"""
    df2 = read_dataframe(path_examples=path_example, data_idx=5)
    model2, future2, forecasted2 = forecast(df2, periods=1096, showflag=True)  # extreme outlieres in June 2015 mess up estimate.

    df2.loc[(df2['ds'] > '2015-06-01') & (df2['ds'] < '2015-06-30'), 'y'] = None
    model2_removed = Prophet().fit(df2)  # Same approach as previous example
    fig = model2_removed.plot(model2_removed.predict(future2))
    fig.set_figheight(18)
    fig.set_figwidth(9)
    plt.title('prediction2 (model with missing data)')
    plt.show()


if __name__ == "__main__":
    main()
