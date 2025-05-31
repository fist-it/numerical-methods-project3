import pandas as pd
from enum import Enum
import numpy as np


def get_canyon_data():
    filename = "./data/WielkiKanionKolorado.csv"
    df = pd.read_csv(filename, sep=",")

    return df


def get_stable_data():
    filename = "./data/SpacerniakGdansk.csv"
    df = pd.read_csv(filename, sep=",")

    return df


def get_unstable_data():
    filename = "./data/rozne_wniesienia.csv"
    df = pd.read_csv(filename, sep=",")

    return df


def get_everest_data():
    filename = "./data/MountEverest.csv"
    df = pd.read_csv(filename, sep=",")

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

    return df, x_max


def get_parsed_tuples(
    type: str, convert: bool = False, step: int = 1, random_step: bool = False
) -> list[tuple[float]]:
    df = None
    match type:
        case "CANYON":
            df = get_canyon_data()
        case "STABLE":
            df = get_stable_data()
        case "UNSTABLE":
            df = get_unstable_data()
        case "EVEREST":
            df = get_everest_data()
        case _:
            raise ValueError("Invalid data type")
    if convert:
        df, x_max = convert_for_precision(df)

    if random_step:
        steps = np.random.randint(1, step + 1, size=len(df))
        indices = np.cumsum(steps)
        indices = indices[indices < len(df)]
        df = df.iloc[indices, :]
    elif step > 1:
        df = df.iloc[::step, :]

    return list(zip(df.iloc[:, 0], df.iloc[:, 1])), x_max if convert else None


def unparse_tuples(
    tuples: list[tuple[float]], x_max: float = None
) -> list[tuple[float]]:
    """
    Convert a list of tuples back from [0, 1] range to original scale if x_max is provided.
    """
    if x_max is not None:
        return [(x * x_max, y) for x, y in tuples]
    return tuples
