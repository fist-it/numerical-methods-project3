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
def third_order_spline_interpolate(
    data: list[tuple[float, float]], x: np.ndarray
) -> np.ndarray:
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
    A = np.zeros((n, n))
    d = np.zeros(n)
    if n < 4:
        raise ValueError(
            "At least 4 data points are required for third-order spline interpolation."
        )
    n = n - 1
    h = np.diff(x_data)
    y = [y_data[i + 1] - y_data[i] / h[i] for i in range(n)]

    A[0, 0] = 1
    A[n, n] = 1

    for i in range(1, n):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        d[i] = 3 * (y[i] - y[i - 1])

    c = np.linalg.solve(A, d)
    a = y_data[:-1]
    b = [(y_data[i + 1] - h[i] * (2 * c[i] + c[i + 1])) / 3 for i in range(n)]
    d = [(c[i + 1] - c[i]) / (3 * h[i]) for i in range(n)]
    c = c[:-1]

    x_new = []
    y_new = []
    for i in range(n):
        x_segment = np.linspace(x_data[i], x_data[i + 1], 100)
        y_segment = (
            a[i]
            + b[i] * (x_segment - x_data[i])
            + c[i] * (x_segment - x_data[i]) ** 2
            + d[i] * (x_segment - x_data[i]) ** 3
        )
        x_new.extend(x_segment)
        y_new.extend(y_segment)

    x_new = np.array(x_new)
    y_new = np.array(y_new)
    np.clip(y_new, 0, None, out=y_new)  # Ensure non-negative values
    np.clip(y_new, None, 10000, out=y_new)  # Ensure y values are within bounds
    return y_new


def test_lagrange():
    step = 20
    data, x_max = get_parsed_tuples(
        "EVEREST", convert=True, step=step, random_step=False
    )
    real_data, _ = get_parsed_tuples(
        "EVEREST", convert=False, step=1, random_step=False
    )
    x = np.linspace(0.0, 1.0, 100)
    y = lagrange_interpolate(data, x)
    x = x * x_max  # Scale x to the original range
    data = unparse_tuples(data, x_max)
    assert len(y) == 100, "Interpolated y values should have length 100"
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label="Lagrange Interpolation", color="green")
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


def test_third_order_spline():
    data, x_max = get_parsed_tuples("EVEREST", convert=True, step=20, random_step=False)
    real_data, _ = get_parsed_tuples(
        "EVEREST", convert=False, step=1, random_step=False
    )
    x = np.linspace(0.0, 1.0, 100)
    y = third_order_spline_interpolate(data, x)
    x = x * x_max  # Scale x to the original range
    data = unparse_tuples(data, x_max)
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label="Third-Order Spline Interpolation", color="green")
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
    plt.title("Third-Order Spline Interpolation of Everest Data")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid()
    save_plot(f"third_order_spline_interpolation_everest.png")
    plt.show()


def main():
    # test_lagrange()
    test_third_order_spline()


if __name__ == "__main__":
    main()
