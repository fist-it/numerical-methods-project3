import numpy as np
from data import get_parsed_tuples
import matplotlib.pyplot as plt


def lagrange_interpolate(data: list[tuple[float, float]], x: np.ndarray) -> np.ndarray:
    """
    Perform Lagrange interpolation on the given data points.

    :param data: List of tuples containing (x, y) data points.
    :param x: List of x values to interpolate.
    :return: List of interpolated y values corresponding to the input x values.
    """
    x_data = np.array([p[0] for p in data])
    y_data = np.array([p[1] for p in data])
    result = np.zeros_like(x, dtype=float)

    print(x_data)
    print(y_data)

    for i in range(len(data)):
        xi = x_data[i]
        li = np.ones_like(x, dtype=float)
        for j in range(len(data)):
            if i != j:
                li *= (x - x_data[j]) / (xi - x_data[j])
        result += y_data[i] * li

    return result


def test_lagrange():
    data, x_max = get_parsed_tuples("EVEREST", convert=True, step=5)
    x = np.linspace(0.0, 1.0, 100)
    y = lagrange_interpolate(data, x)
    assert len(y) == 100, "Interpolated y values should have length 100"
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label="Lagrange Interpolation", color="green")
    plt.scatter(
        [p[0] for p in data],
        [p[1] for p in data],
        label="Original Data",
        color="blue",
        alpha=0.5,
    )
    plt.title("Lagrange Interpolation of Everest Data")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid()
    plt.show()


def main():
    test_lagrange()


if __name__ == "__main__":
    main()
