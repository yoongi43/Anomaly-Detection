from prophet import Prophet
import matplotlib.pyplot as plt


def forecast(df, periods, showflag=True):
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=periods)
    predict = m.predict(future)

    if showflag:
        fig = m.plot(predict)
        fig.set_figheight(18)
        fig.set_figwidth(9)
        plt.title("forcasted")
        plt.show()

    return m, future, predict