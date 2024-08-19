import json
from tqdm import tqdm
from ultralytics import FastSAM
from segment_anything import sam_model_registry, SamPredictor
from utilities import *
import traceback


def fast_sam(name: str, iou: float = 0.1, conf: float = 0.4, image_h: int = 692,
             image_w: int = 1029, stream: bool = True) -> None:
    """
    Detect objects and saves bounding boxes.

    Args:
    - name (str): File name.

    Returns:
    - None
    """

    # Create a FastSAM model
    model = FastSAM("weights/FastSAM-s.pt")  # FastSAM-x.pt # For more accurate results. takes 10X time per image.

    results = model.track(source=f"../inter/videos/{name}.mp4", line_width=1, save=False, show=False,
                          tracker="bytetrack.yaml", iou=iou, conf=conf, imgsz=[image_h, image_w], stream=stream)

    for i, result in enumerate(results):
        os.makedirs(f'../inter/json_files/{name}', exist_ok=True)
        json_data = result.tojson()

        with open(f'../inter/json_files/{name}/{i}.json', 'w') as fl:
            fl.write(json_data)

    return None


def read_json_files(json_dir: str) -> list:
    """
    Reads and sorts JSON files from a directory.

    Args:
    - json_dir (str): Path to the directory containing JSON files.

    Returns:
    - list: Sorted list of JSON file paths.
    """
    json_paths = os.listdir(json_dir)
    return sorted(json_paths, key=lambda x: int(os.path.splitext(x)[0]))


def setup_video_io(name: str) -> tuple:
    """
    Sets up video capture and writer objects.

    Args:
    - name (str): File name.

    Returns:
    - tuple: VideoCapture object, VideoWriter object, frame width, frame height, and fps.
    """
    vid_path = f'../inter/videos/{name}.mp4'
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        raise IOError(f"Error: Could not open video file {vid_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f'../data/videos/{name}_seg.mp4', fourcc, fps, (width, height), isColor=True)

    return cap, out, width, height, fps


def initialize_sam_predictor(device) -> SamPredictor:
    """
    Initializes and returns a SAM predictor.

    Returns:
    - SamPredictor: Initialized SAM predictor.
    """
    sam_checkpoint = "weights/SAM_B_model.pth"
    model_type = "vit_b"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    return SamPredictor(sam)


def process_frame(frame: np.ndarray, predictor: SamPredictor, frame_data: list, dp_threshold: int) -> tuple:
    """
    Processes a single frame, generating masks and splines for objects.

    Args:
    - frame (np.ndarray): Video frame.
    - predictor (SamPredictor): SAM predictor object.
    - frame_data (list): List of object data for the frame.

    Returns:
    - tuple: Processed frame, dictionary of object areas, and dictionary of anchor and spline points.
    """
    predictor.set_image(frame)
    frame_areas = {}
    frame_points = {}

    for item in frame_data:
        input_box = np.array([item["box"]["x1"], item["box"]["y1"], item["box"]["x2"], item["box"]["y2"]])
        masks, _, _ = predictor.predict(point_coords=None, point_labels=None, box=input_box[None, :],
                                        multimask_output=False)

        obj_mask = masks.squeeze()
        contours, _ = cv2.findContours(obj_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            largest_contour = largest_contour.reshape(-1, 2)
            anchor_points = douglas_peucker(largest_contour, len(largest_contour) // dp_threshold)
            spline_points = get_spline_points(anchor_points, smoothing=1, power=3)

            if spline_points is not None:
                ID_number = item['track_id']
                area = calculate_polygon_area(spline_points)
                frame_areas[f"obj_{ID_number}"] = area
                frame_points[f"obj_{ID_number}"] = {
                    "anchor_points": anchor_points.tolist(),
                    "spline_points": spline_points.tolist()
                }

                frame = draw_on_frame(frame, spline_points, input_box, ID_number)

    return frame, frame_areas, frame_points


def draw_on_frame(frame: np.ndarray, spline_points: np.ndarray, input_box: np.ndarray, ID_number: int) -> np.ndarray:
    """
    Draws splines, bounding boxes, and ID numbers on the frame.

    Args:
    - frame (np.ndarray): Video frame.
    - spline_points (np.ndarray): Points defining the spline.
    - input_box (np.ndarray): Bounding box coordinates.
    - ID_number (int): Object ID number.

    Returns:
    - np.ndarray: Frame with drawings.
    """
    for k in range(len(spline_points) - 1):
        start, end = tuple(np.int32(spline_points[k])), tuple(np.int32(spline_points[k + 1]))
        cv2.line(frame, start, end, color=(0, 0, 0), thickness=1)

    cv2.rectangle(frame, (int(input_box[0]), int(input_box[1])), (int(input_box[2]), int(input_box[3])),
                  color=(255, 255, 255), thickness=1)

    M = cv2.moments(spline_points)
    if M["m00"] != 0:
        cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        cv2.putText(frame, f'{ID_number}', (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return frame


def update_areas_dict(dict_areas: dict, frame_areas: dict, frame_index: int, num_images: int) -> dict:
    """
    Updates the dictionary of object areas with data from the current frame.

    Args:
    - dict_areas (dict): Dictionary of object areas across all frames.
    - frame_areas (dict): Dictionary of object areas for the current frame.
    - frame_index (int): Current frame index.
    - num_images (int): Total number of frames.

    Returns:
    - dict: Updated dictionary of object areas.
    """
    for key, area in frame_areas.items():
        if key not in dict_areas:
            dict_areas[key] = [None] * num_images
        dict_areas[key][frame_index] = area
    return dict_areas


def save_areas_to_csv(dict_areas: dict, save_path: str) -> None:
    """
    Saves the areas data to a CSV file.

    Args:
    - dict_areas (dict): Dictionary of object areas across all frames.
    - save_path (str): Path to save the CSV file.

    Returns:
    - None
    """
    df = pd.DataFrame(dict_areas)
    df.to_csv(save_path)
    print(f"\nSaved DataFrame at {save_path}\n")


def vid_sam(name: str, device: str = "cpu", dp_threshold: int = 12) -> None:
    """
    Uses bounding boxes from FastSAM to generate masks within them. Then fits a spine around its contours,
    arranges area data in a data frame and saves the video with the object tracking info on top.

    Args:
    - name (str): File name.

    Returns:
    - None
    """
    json_dir = f'../inter/json_files/{name}'
    json_paths = read_json_files(json_dir)
    num_images = len(json_paths)

    cap, out, width, height, fps = setup_video_io(name)
    predictor = initialize_sam_predictor(device)

    dict_areas = {}
    dict_points = {}

    for i, jp in tqdm(enumerate(json_paths), total=num_images, colour='green', ascii=True):
        try:
            with open(f'{json_dir}/{jp}', 'r') as f:
                frame_data = json.load(f)

        except Exception as e:
            print(f"\nError reading JSON file {jp}.\n{e}\n")
            continue

        ret, frame = cap.read()
        if not ret:
            print(f"\nError reading frame {i} from video.\n")
            continue

        try:
            processed_frame, frame_areas, frame_points = process_frame(frame, predictor, frame_data, dp_threshold)
            dict_areas = update_areas_dict(dict_areas, frame_areas, i, num_images)
            dict_points[i] = frame_points
            out.write(processed_frame)
        except Exception as e:
            print(f"\nError processing frame {i}.\n{e}\n")
            traceback.print_exc()
            continue

        if not out.isOpened():
            print(f"Error: VideoWriter closed unexpectedly at frame {i}")
            break

    cap.release()
    out.release()

    save_areas_to_csv(dict_areas, f"../data/areas_csv/{name}_SAM.csv")

    # Save points data to a JSON file
    with open(f"../data/points_json/{name}_points.json", "w") as f:
        json.dump(dict_points, f)
    import pickle
    with open(f"../data/points_json/{name}_points.pkl", "wb") as f:
        pickle.dump(dict_points, f, protocol=pickle.HIGHEST_PROTOCOL)

    return None


import os
import json
import gzip

json_file_path = '/Users/ypm/Desktop/Track/data/points_json/potato_points.json'
compressed_json_file_path = '/Users/ypm/Desktop/Track/data/points_json/potato_points.json.gz'

with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

with gzip.open(compressed_json_file_path, 'wt', encoding='utf-8') as zipfile:
    json.dump(data, zipfile)
