import numpy as np
from data import get_parsed_tuples, unparse_tuples
import matplotlib.pyplot as plt
from plot import save_plot


def lagrange_interpolate(
    data: list[tuple[float, float]],
    x: np.ndarray,
    clip_min: int = 0,
    clip_max: int = 10000,
) -> np.ndarray:
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

    np.clip(result, clip_min, clip_max, out=result)  # Ensure non-negative values
    return result


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

    n = len(data) - 1
    A = np.zeros((4 * n, 4 * n), dtype=np.float64)
    b = np.zeros(4 * n, dtype=np.float64)
    b[::4] = y_data[:-1]
    b[1::4] = y_data[1:]

    for i in range(0, n):
        h = x_data[i + 1] - x_data[i]
        A[4 * i, 4 * i] = 1  # a_i

        A[4 * i + 1, 4 * i + 0] = h**0
        A[4 * i + 1, 4 * i + 1] = h**1
        A[4 * i + 1, 4 * i + 2] = h**2
        A[4 * i + 1, 4 * i + 3] = h**3

        if i < n - 1:
            A[4 * i + 2, 4 * i + 1] = 1
            A[4 * i + 2, 4 * i + 2] = 2 * h
            A[4 * i + 2, 4 * i + 3] = 3 * h**2
            A[4 * i + 2, 4 * i + 5] = -1

            A[4 * i + 3, 4 * i + 2] = 2
            A[4 * i + 3, 4 * i + 3] = 6 * h
            A[4 * i + 3, 4 * i + 6] = -2

    A[-2, 2] = 1  # Set the last point's derivative to 0
    A[-1, -2] = 2
    A[-1, -1] = 6 * (
        x_data[-1] - x_data[-2]
    )  # Set the last point's second derivative to 0

    coeffs = np.linalg.solve(A, b).flatten()
    coeffs_n = 0
    result = np.zeros_like(x, dtype=float)

    for i, point in enumerate(x):
        while coeffs_n < len(x_data) - 2 and not (
            x_data[coeffs_n] <= point <= x_data[coeffs_n + 1]
        ):
            coeffs_n += 1

        h = point - x_data[coeffs_n]
        result[i] = (
            coeffs[4 * coeffs_n]
            + coeffs[4 * coeffs_n + 1] * h
            + coeffs[4 * coeffs_n + 2] * h**2
            + coeffs[4 * coeffs_n + 3] * h**3
        )

    return result


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
    data, x_max = get_parsed_tuples("CANYON", convert=True, step=20, random_step=False)
    real_data, _ = get_parsed_tuples("CANYON", convert=False, step=1, random_step=False)
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
    # save_plot(f"third_order_spline_interpolation_everest.png")
    plt.show()
    plt.close()


def get_interpolation_types() -> list[str]:
    """
    Get the list of available interpolation types.
    """
    return ["Lagrange", "Spline"]
    # return ["Spline"]


def main():
    # test_lagrange()
    test_third_order_spline()


if __name__ == "__main__":
    main()
