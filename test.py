import data
from data import DataType
import matplotlib.pyplot as plt

FIGURE_DIRECTORY = "./figures/"


def test_random_step():
    coordinates: list[tuple[float, float]] = data.get_parsed_tuples(DataType.EVEREST,
                                                                    convert=True)

    coordinates_random_step: list[tuple[float, float]] = data.get_parsed_tuples(DataType.EVEREST,
                                                                                convert=True,
                                                                                step=5,
                                                                                random_step=True)

    x_start = 0.2
    x_end = 0.4

    coordinates_slice = [
        coord for coord in coordinates[0] if x_start <= coord[0] <= x_end]
    coordinates_random_step_slice = [
        coord for coord in coordinates_random_step[0] if x_start <= coord[0] <= x_end]

    x = [coord[0] for coord in coordinates_slice]
    y = [coord[1] for coord in coordinates_slice]

    x_random_step = [coord[0] for coord in coordinates_random_step_slice]
    y_random_step = [coord[1] for coord in coordinates_random_step_slice]

    plt.figure(figsize=(10, 6))

    plt.scatter(x, y, label="Original Data", color='blue', alpha=0.2)
    plt.scatter(x_random_step, y_random_step,
                label="Random Step Data", color='red', alpha=0.8)
    plt.title("Everest")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    plt.savefig(FIGURE_DIRECTORY + "random_step.png")


def print_slice(data_list, start, end):
    """
    Print a slice of the data list from start to end index.
    """
    for index, item in enumerate(data_list[start:end], start=start):
        print(f"{index}: {item}")


def test_conversion():
    analyzed_data = data.get_stable_data()
    assert analyzed_data is not None, "Data should not be None"

    analyzed_data, _ = data.convert_for_precision(analyzed_data)
    assert analyzed_data is not None, "Converted data should not be None"


def test_parsed_tuples():
    type = DataType.CANYON
    analyzed_data = data.get_parsed_tuples(type)
    assert analyzed_data is not None, "Parsed tuples should not be None"
    assert len(analyzed_data) > 0, "Parsed tuples should not be empty"
    print_slice(analyzed_data, 10, 16)

    # Test with converted data
    type = DataType.EVEREST
    analyzed_data = data.get_parsed_tuples(type, convert=True)
    assert analyzed_data is not None, "Parsed tuples should not be None"
    assert len(analyzed_data) > 0, "Parsed tuples should not be empty"
    print_slice(analyzed_data, 10, 16)


def main():
    # create figures/ directory if it doesn't exist
    import os
    if not os.path.exists(FIGURE_DIRECTORY):
        os.makedirs(FIGURE_DIRECTORY)
    test_conversion()
    test_parsed_tuples()
    test_random_step()
    print("All tests passed!")


if __name__ == "__main__":
    main()
