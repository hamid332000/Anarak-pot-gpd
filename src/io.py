import pandas as pd

def read_temperature_data(filename,
                          date_column=0,
                          value_column=1):

    df = pd.read_excel(filename)

    df = df.iloc[:, [date_column, value_column]]

    df.columns = ["Date", "Tmax"]

    df["Date"] = pd.to_datetime(df["Date"])

    df["Tmax"] = pd.to_numeric(df["Tmax"])

    return df