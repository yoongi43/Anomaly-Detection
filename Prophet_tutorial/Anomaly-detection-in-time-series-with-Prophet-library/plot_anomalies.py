import altair as alt


def plot_anomalies(forecasted):
    interval = alt.Chart(forecasted).mark_area(interpolate="basis", color='#7FC97F').encode(
    x=alt.X('ds:T', title ='date'),
    y='yhat_upper',
    y2='yhat_lower',
    tooltip=['ds', 'fact', 'yhat_lower', 'yhat_upper']
    ).interactive().properties(
        title='Anomaly Detection'
    )

    fact = alt.Chart(forecasted[forecasted.anomaly == 0]).mark_circle(size=15, opacity=0.7, color='Black').encode(
        x='ds:T',
        y=alt.Y('fact', title='sales'),
        tooltip=['ds', 'fact', 'yhat_lower', 'yhat_upper']
    ).interactive()

    anomalies = alt.Chart(forecasted[forecasted.anomaly != 0]).mark_circle(size=30, color='Red').encode(
        x='ds:T',
        y=alt.Y('fact', title='sales'),
        tooltip=['ds', 'fact', 'yhat_lower', 'yhat_upper'],
        size=alt.Size('importance', legend=None)
    ).interactive()

    return alt.layer(interval, fact, anomalies)\
              .properties(width=870, height=450)\
              .configure_title(fontSize=20)
