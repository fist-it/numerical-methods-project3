import pandas as pd


def get_canyon_data():
    filename = './data/WielkiKanionKolorado.csv'
    df = pd.read_csv(filename, sep=',')

    return df


def get_stable_data():
    filename = './data/SpacerniakGdansk.csv'
    df = pd.read_csv(filename, sep=',')

    return df


def get_unstable_data():
    filename = './data/rozne_wniesienia.csv'
    df = pd.read_csv(filename, sep=',')

    return df


def get_everest_data():
    filename = './data/MountEverest.csv'
    df = pd.read_csv(filename, sep=',')

    return df


def convert_for_precision(df):
    """
    Convert the DataFrame to a format suitable for precision analysis.
    """
    x_max = df.iloc[:, 0].iloc[-1]
    new_x = [0] * len(df)
    for i in range(len(df)):
        new_x[i] = df.iloc[i, 0] / x_max

    df.iloc[:, 0] = new_x

    return df
