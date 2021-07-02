# function to load data and return as inputs and output


def load_data(filename: str = None, output: str = None, separator: str = ','):

    valid_sep = {",", "\t", " "}
    if separator not in valid_sep:
        raise ValueError("separator must be one of %r." % valid_sep)

    if filename and output is not None:
        import pandas as pd
        import numpy as np

        tmp_data = pd.read_csv("./"+filename, sep=separator, dtype=np.float64)
        data_input = tmp_data.drop(output, axis=1)
        data_output = tmp_data[output]

    return [data_input, data_output]
