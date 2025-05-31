import numpy as np
from data import get_parsed_tuples, unparse_tuples
import matplotlib.pyplot as plt
from plot import save_plot


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

    # print(x_data)
    # print(y_data)

    for i in range(len(data)):
        xi = x_data[i]
        li = np.ones_like(x, dtype=float)
        for j in range(len(data)):
            if i != j:
                li *= (x - x_data[j]) / (xi - x_data[j])
        result += y_data[i] * li

    np.clip(result, 0, None, out=result)  # Ensure non-negative values
    np.clip(result, None, 10000, out=result)
    return result


# TODO: Implement third-order spline interpolation
def third_order_spline_interpolate(data: list[tuple[float, float]], x: np.ndarray) -> np.ndarray:
    """
    Perform third-order spline interpolation on the given data points.
    Pure, not-wrapper function.

    :param data: List of tuples containing (x, y) data points.
    :param x: List of x values to interpolate.
    :return: List of interpolated y values corresponding to the input x values.
    """
    x_data = np.array([p[0] for p in data])
    y_data = np.array([p[1] for p in data])
    n = len(data)


def test_lagrange():
    step = 20
    data, x_max = get_parsed_tuples(
        "EVEREST", convert=True, step=step, random_step=False
    )
    real_data, _ = get_parsed_tuples(
        "EVEREST", convert=False, step=1, random_step=False
    )
    # print(real_data[-1])
    # print(data[-1])
    x = np.linspace(0.0, 1.0, 100)
    y = lagrange_interpolate(data, x)
    x = x * x_max  # Scale x to the original range
    data = unparse_tuples(data, x_max)
    assert len(y) == 100, "Interpolated y values should have length 100"
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label="Lagrange Interpolation", color="green")
    # print(p for p in real_data)
    plt.scatter(
        [p[0] for p in real_data],
        [p[1] for p in real_data],
        label="Real Data",
        color="red",
        alpha=0.5,
    )
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
    save_plot(f"lagrange_interpolation_everest_{step}_step.png")
    plt.show()


def main():
    test_lagrange()


if __name__ == "__main__":
    main()
