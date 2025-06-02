import data
import numpy as np
import matplotlib.pyplot as plt
from interpolation import (
    lagrange_interpolate,
    third_order_spline_interpolate,
    get_interpolation_types,
)
from plot import save_plot

POINTS_COUNT = [2, 4, 8, 16, 32, 64, 128, 256]


def demonstrate_interpolation(
    point_count: int, interpolation_type: str, data_type, ax=None
) -> None:
    current_data, x_max = data.get_parsed_tuples(data_type, True)
    # get n evenly spaced points (index wise) from everest_data x's
    length = len(current_data)
    # get only 8 evenly spaced points
    x_indices = np.linspace(0, length - 1, point_count, dtype=int)
    # trim the everest_data to only include the x's in x_indices
    chosen_points = [current_data[i] for i in x_indices]

    temp_data, _ = data.get_parsed_tuples(data_type, True)
    x = [i[0] for i in temp_data]
    if interpolation_type == "Lagrange":
        y = lagrange_interpolate(chosen_points, x)
    elif interpolation_type == "Spline":
        y = third_order_spline_interpolate(chosen_points, x)

    interpolated = data.unparse_tuples(list(zip(x, y)), x_max)
    chosen_points = data.unparse_tuples(chosen_points, x_max)

    current_data = data.unparse_tuples(current_data, x_max)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        [point[0] for point in current_data],
        [point[1] for point in current_data],
        color="red",
        label="Original Data",
        alpha=0.6,
    )
    ax.plot(
        [point[0] for point in interpolated],
        [point[1] for point in interpolated],
        color="blue",
        label=f"{interpolation_type} Interpolation",
    )
    ax.scatter(
        [point[0] for point in chosen_points],
        [point[1] for point in chosen_points],
        color="green",
        label="Chosen Points",
        edgecolor="black",
    )
    ax.set_title(f"{interpolation_type} Interpolation of {data_type} Data")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.legend()
    ax.grid()
    # save_plot(
    #     f"{interpolation_type}_interpolation_of_{data_type}_{point_count}_points.png"
    # )


def main():
    data_types = data.get_data_types()
    interpolation_types = get_interpolation_types()
    for data_type in data_types:
        for interpolation_type in interpolation_types:
            print(
                f"Demonstrating {interpolation_type} interpolation for {data_type} data"
            )

            # Process in groups of 4
            for i in range(0, len(POINTS_COUNT), 4):
                group = POINTS_COUNT[i : i + 4]
                rows, cols = 2, 2
                fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
                axes = axes.flatten()  # easier indexing

                for j, point_count in enumerate(group):
                    demonstrate_interpolation(
                        point_count, interpolation_type, data_type, ax=axes[j]
                    )

                fig.tight_layout()
                save_plot(f"{interpolation_type}_{data_type}_group_{i // 4 + 1}.png")
                plt.close(fig)  # free memory between plots

    return None


if __name__ == "__main__":
    main()
