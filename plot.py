import matplotlib.pyplot as plt
import numpy as np
import data
from data import DataType

from pathlib import Path

PLOT_DIRECTORY = Path("plots/")


def plot_everest():
    coordinates: list[tuple(float)] = data.get_parsed_tuples(
        DataType.EVEREST, convert=True
    )

    coordinates_step: list[tuple(float)] = data.get_parsed_tuples(
        DataType.EVEREST, convert=True, step=2
    )

    coordinates_random_step: list[tuple(float)] = data.get_parsed_tuples(
        DataType.EVEREST, convert=True, step=5, random_step=True
    )
    plt.figure(figsize=(10, 6))
    x = [coord[0] for coord in coordinates]
    y = [coord[1] for coord in coordinates]
    x_step = [coord[0] for coord in coordinates_step]
    y_step = [coord[1] for coord in coordinates_step]

    x_random_step = [coord[0] for coord in coordinates_random_step]
    y_random_step = [coord[1] for coord in coordinates_random_step]

    plt.plot(x, y, marker="o", linestyle="-", color="b", markersize=5)
    plt.plot(x_step, y_step, marker="o", linestyle="-", color="r", markersize=5)
    plt.plot(
        x_random_step, y_random_step, marker="o", linestyle="-", color="g", markersize=5
    )
    plt.title("Everest")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    save_plot(plt, "everest_plot.png")
    plt.show()


def save_plot(filename: str):
    """
    Save the plot to the specified filename in the PLOT_DIRECTORY.
    """
    PLOT_DIRECTORY.mkdir(exist_ok=True)
    filepath = PLOT_DIRECTORY / filename
    plt.savefig(filepath, dpi=300)
    print(f"Plot saved to {filepath}")
