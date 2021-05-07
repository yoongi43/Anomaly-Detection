import pandas as pd
import os
import matplotlib.pyplot as plt
from read_dataframe import read_dataframe
from fit_predict_model import fit_predict_model
from detect_anomalies import detect_anomalies
from plot_anomalies import plot_anomalies
import altair_viewer as altv


def main():
    path_examples = os.path.abspath("../examples")
    df = read_dataframe(path_examples, data_idx=2)

    pred = fit_predict_model(df)
    pred = detect_anomalies(pred)
    plot_anomalies(pred)


if __name__=="__main__":
    main()