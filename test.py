from data import *


def test_conversion():
    data = get_stable_data()
    assert data is not None, "Data should not be None"

    data = convert_for_precision(data)
    assert data is not None, "Converted data should not be None"

    print(data)


def main():
    test_conversion()
    print("All tests passed!")


if __name__ == "__main__":
    main()
