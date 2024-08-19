import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np
import pandas as pd
import matplotlib.cm as cm


def spline_interp(x, y, N, k=3):
    """
    Wrapper function for interpolating between points using a polynomial
    spline representation of the points. Linear interpolation is done for k=1.

    Parameters
    ----------
    x : list of floats
        x coordinates of points to be interpolated
    y : list of floats
        y coordinates of points to be interpolated
    N : int
        Number of interpolated coordinates to return
    k : int, optional
        Degree of polynomial to use for interpolation. The default is 3.

    Returns
    -------
    xInt : list of floats
        interpolated x coordinates
    yInt : list of floats
        interpolated y coordinates

    """
    # Check to see if the first point is equal to the last point.
    pointsEqual = x[0] == x[-1] and y[0] == y[-1]
    # If they are not the same, add the first point to the end
    if not pointsEqual:
        x.append(x[0])
        y.append(y[0])

    # Create a cubic spline representation (with periodic boundary conditions)
    tck, u = interpolate.splprep([x, y], k=3, s=0, per=1, quiet=1)

    # Define range for interpolation between handles and interpolate
    unew = np.linspace(0, 1.0, N)
    xInt, yInt = interpolate.splev(unew, tck)

    return xInt, yInt


def get_area(points, nInterp):
    """
    Use the Shoelace formula as implemented on Stack Overflow at this link:
    https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
    to get the area of a polygon. Note that for accuracy, this should be done
    using the cubic spline interpolated points rather than the handles of
    the ROI.
    """
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    xInt, yInt = spline_interp(x, y, nInterp)

    area = 0.5 * np.abs(np.dot(xInt, np.roll(yInt, 1)) - np.dot(yInt, np.roll(xInt, 1)))
    return area


import csv


def save_csv(data):
    # Get all unique frame numbers and granule numbers
    frame_numbers = sorted(data.keys())
    granule_numbers = sorted(
        set(granule for frame in data.values() for granule in frame.keys() if isinstance(granule, int)))

    # Prepare the CSV data
    csv_data = [['Index'] + granule_numbers]
    for frame in frame_numbers:
        row = [frame]
        for granule in granule_numbers:
            if granule in data[frame]:
                area = get_area(data[frame][granule], 100)
                row.append(area)
            else:
                row.append('')
        csv_data.append(row)

    # Write to CSV file
    with open("../data/areas_csv/granules.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_data)


import pickle


def read_pickle(file_path: str) -> None:
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    save_csv(data)


# read_pickle("/Users/ypm/Desktop/SAM2Shot/data/PotatoPotat.pkl")

def plot_csv_data() -> None:
    """
    Plots all curves in one plot with different shades of the same color.

    Args:
    - name (str): File name

    Returns:
    - None
    """

    df = pd.read_csv(f"../data/areas_csv/granule_areas.csv")
    columns = df.columns

    # df_nan = pd.DataFrame(columns=df.columns)
    # for index, row in df.iterrows():
    #     df_nan = df_nan._append(row, ignore_index=True)
    #     df_nan = df_nan._append(pd.Series([np.nan] * len(df.columns), index=df.columns), ignore_index=True)
    # df = df_nan.reset_index(drop=True)

    plt.figure(figsize=(10, 6))

    cmap = cm.get_cmap("Blues", len(columns))
    for i, column in enumerate(columns):
        plt.scatter(range(len(df[column])), df[column], color="blue", alpha=0.5, label=column, s=5)

    # ------------------------------------------------------------------------------

    df = pd.read_csv(f"../data/areas_csv/potato_SAM.csv")
    columns = df.columns

    # threshold = 0.05
    # valid_columns = [col for col in columns if df[col].isna().mean() < threshold]
    # print(len(valid_columns))
    selected_columns = df[['obj_32', 'obj_3', 'obj_2']]

    cmap = cm.get_cmap("Reds", len(selected_columns))
    for i, column in enumerate(selected_columns):
        plt.scatter(range(len(df[column])), df[column], color='red', alpha=0.5, label=column, s=5)

    plt.title('Index vs Areas')
    plt.xlabel('Index')
    plt.ylabel('Areas')
    plt.grid(True)
    plt.ylim(0, 15000)
    plt.tight_layout()

    plt.show()


plot_csv_data()
