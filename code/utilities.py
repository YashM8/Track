import os
import glob
from PIL import Image
import numpy as np
from scipy.interpolate import splprep, splev
from shapely.geometry import Polygon
import pandas as pd
import matplotlib.pyplot as plt
import cv2


def process_gray_resize(folder_path, output_folder, skip, scale_down):
    """
    Processes images by converting to grayscale, resizing, and saving as PNG.

    Parameters:
    - folder_path: Path to the folder containing input images.
    - output_folder: Path where resized grayscale images will be saved.
    """
    folder_path, output_folder = os.path.abspath(folder_path), os.path.abspath(output_folder)

    if not os.path.exists(output_folder): os.makedirs(output_folder)

    image_files = sorted(
        glob.glob(os.path.join(folder_path, 'recording_*.*')),
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[1]), reverse=True)

    if not image_files: raise NameError("\nIncorrect folder path\n")

    for i, image_file in enumerate(image_files[::skip]):
        img = Image.open(image_file).convert('L')
        width, height = img.size
        img = img.resize((width // scale_down, height // scale_down))
        output_file_path = os.path.join(output_folder, f'image_{i + 1:04d}.png')
        img.save(output_file_path)
        if i % 10 == 0:
            print("Processing...")

    print("\n---- Done ----\n")


def douglas_peucker(points, epsilon):
    """
    Simplify a polygon using the Douglas-Peucker algorithm.

    Args:
    - points (numpy.ndarray): Points defining the polygon.
    - epsilon (float): Tolerance parameter for simplification.

    Returns:
    - simplified_points (numpy.ndarray): Simplified points of the polygon.
    """
    if len(points) <= 2:
        return points
    d_max = 0
    index = 0
    end = len(points) - 1
    for i in range(1, end):
        d = perpendicular_distance(points[i], points[0], points[end])
        if d > d_max:
            index = i
            d_max = d
    if d_max > epsilon:
        result1 = douglas_peucker(points[:index + 1], epsilon)
        result2 = douglas_peucker(points[index:], epsilon)
        result = np.vstack((result1[:-1], result2))
    else:
        result = np.array([points[0], points[end]])
    return result


def perpendicular_distance(point, line_start, line_end):
    """
    Calculate perpendicular distance from a point to a line segment.

    Args:
    - point (tuple): Coordinates of the point.
    - line_start (tuple): Coordinates of the start of the line segment.
    - line_end (tuple): Coordinates of the end of the line segment.

    Returns:
    - distance (float): Perpendicular distance from the point to the line segment.
    """
    x0, y0 = point
    x1, y1 = line_start
    x2, y2 = line_end
    return abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)


def get_spline_points(anchor_points, smoothing, power):
    """
    Generate spline points from anchor points using interpolation.

    Args:
    - anchor_points (numpy.ndarray): Anchor points of the spline.

    Returns:
    - spline_points (numpy.ndarray): Points defining the spline curve.
    """
    try:
        tck, u = splprep(anchor_points.T, s=smoothing, per=True, k=power)
        u_new = np.linspace(u.min(), u.max(), 400)
        x_new, y_new = splev(u_new, tck, der=0)

        spline_points = np.vstack((x_new, y_new)).T
        spline_points = np.round(spline_points).astype(int)
    except Exception as e:
        print(e)
        return None
    return spline_points


def calculate_polygon_area(points):
    """
    Calculate the area of a polygon defined by points.

    Args:
    - points (numpy.ndarray): Points defining the polygon.

    Returns:
    - area (float): Area of the polygon.
    """
    if points is None:
        return None
    return Polygon(points).area


def plot_csv_data(name: str, show: bool) -> None:
    """
    Plots the Index vs Areas in pixels.

    Args:
    - name (str): File name

    Returns:
    - None
    """
    df = pd.read_csv(f"../data/areas_csv/{name}_SAM.csv")
    columns = df.columns
    num_columns = len(columns)
    n_plots = 3
    n_rows = (num_columns + n_plots - 1) // n_plots

    colors = plt.cm.get_cmap('tab10', len(columns))
    fig, axes = plt.subplots(n_rows, n_plots, figsize=(15, 3 * n_rows), constrained_layout=True)
    axes = axes.flatten()

    for i, column in enumerate(columns):
        ax = axes[i]
        ax.scatter(df.index, df[column], marker='o', s=15, alpha=1, color=colors(i))
        ax.set_title(column)
        ax.set_xlabel('Index')
        ax.set_ylabel('Areas')

    for ax in axes[num_columns:]:
        ax.axis('off')

    dpi = 250
    plt.savefig(f"../figures/{name}_areas.png", dpi=dpi)
    if show:
        plt.show(dpi=dpi)


def process_gray_resize_to_video(folder_path, output_video_path, skip, fps=30, scale_down=3):
    """
    Processes images by converting to grayscale, resizing, and creating a video.

    Parameters:
    - folder_path: Path to the folder containing input images.
    - output_video_path: Path where the output video will be saved.
    - skip: Number of frames to skip between each processed frame.
    - fps: Frames per second for the output video.
    """

    image_files = sorted(
        glob.glob(os.path.join(folder_path, '*.*')),
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[1]) if '_' in os.path.basename(x) and
                                                                                     os.path.basename(
                                                                                         x).lower().endswith(('.png',
                                                                                                              '.jpg',
                                                                                                              '.jpeg')) else -1,
        reverse=True
    )

    if not image_files:
        raise NameError("No images found. Please check the directory path and file naming convention.")

    first_img = Image.open(image_files[0]).convert('L')
    width, height = first_img.size
    new_width, new_height = width // scale_down, height // scale_down

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    os.makedirs(f'../inter/videos/{os.path.basename(output_video_path[-4])}', exist_ok=True)
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (new_width, new_height), isColor=False)

    for i, image_file in enumerate(image_files[::skip]):
        img = Image.open(image_file).convert('L')
        img = img.resize((new_width, new_height))

        # Convert PIL image to OpenCV format
        frame = np.array(img)

        clahe = cv2.createCLAHE(clipLimit=10, tileGridSize=(8, 8))
        clahe_frame = clahe.apply(frame)

        out.write(clahe_frame)

        if i % 10 == 0:
            print("Processing ...")

    out.release()
    print(f"Video saved as {output_video_path}")
    print("\n---- Done ----\n")


def get_gradient_magnitudes(contour):
    gradients = np.gradient(contour[:, 1], contour[:, 0])

    magnitudes = np.abs(gradients)  #np.sqrt(gradients[0] ** 2 + gradients[1] ** 2)
    return magnitudes


def find_highest_gradient_points(contour, num_points):
    magnitudes = get_gradient_magnitudes(contour)
    highest_indices = np.argsort(-magnitudes)[:num_points]

    return contour[highest_indices]


def simplify_contour(contour):
    actual_area = calculate_polygon_area(contour)

    contour = contour[::3]
    start_points = 4
    spline_points = contour
    while True and start_points > 15:
        anchor_points = find_highest_gradient_points(contour, start_points)
        if len(anchor_points) < 3:
            start_points += 1
            continue
        spline_points = get_spline_points(anchor_points, smoothing=1, power=3)
        spline_area = calculate_polygon_area(spline_points)

        if np.abs(spline_area - actual_area) / actual_area <= 0.1:
            break

        start_points += 1

    return spline_points


from scipy.interpolate import CubicSpline
from scipy.optimize import minimize


def douglas_peucker_spline(points, epsilon):
    """
    Simplify a polygon using the Douglas-Peucker algorithm with cubic spline.

    Args:
    - points (numpy.ndarray): Points defining the polygon.
    - epsilon (float): Tolerance parameter for simplification.

    Returns:
    - simplified_points (numpy.ndarray): Simplified points of the polygon.
    """
    if len(points) <= 2:
        return points
    d_max = 0
    index = 0
    end = len(points) - 1

    # Fit cubic spline
    cs = CubicSpline(np.arange(len(points)), points, bc_type='clamped')

    # Calculate perpendicular distances
    for i in range(1, end):
        d = max_distance_to_spline(points[i], cs)
        if d > d_max:
            index = i
            d_max = d

    if d_max > epsilon:
        result1 = douglas_peucker_spline(points[:index + 1], epsilon)
        result2 = douglas_peucker_spline(points[index:], epsilon)
        result = np.vstack((result1[:-1], result2))
    else:
        result = np.array([points[0], points[end]])

    return result


def max_distance_to_spline(point, spline):
    """
    Calculate the maximum perpendicular distance from a point to a cubic spline.

    Args:
    - point (numpy.ndarray): Coordinates of the point.
    - spline (CubicSpline): Cubic spline object.

    Returns:
    - distance (float): Maximum perpendicular distance from the point to the cubic spline.
    """

    def distance(x):
        spline_point = spline(x)
        return np.linalg.norm(spline_point - point)

    result = minimize(distance, x0=np.array([0]), bounds=[(0, 1)])
    return result.fun