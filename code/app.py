import os
from flask import Flask, render_template, jsonify, request, send_file
import cv2
import json
import io
from utilities import calculate_polygon_area
import csv
import pandas as pd

app = Flask(__name__)

VIDEO_FOLDER = "/Users/ypm/Desktop/Track/inter/videos"


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_list')
def video_list():
    videos = [f.split('.')[0] for f in os.listdir(VIDEO_FOLDER) if f.endswith('.mp4')]
    print(videos)
    return jsonify(videos)


@app.route('/video/<name>')
def video_info(name):
    video_path = f'../inter/videos/{name}.mp4'
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    points_path = f'../data/points_json/{name}_points.json'
    with open(points_path, 'r') as f:
        points_data = json.load(f)

    return jsonify({
        'frame_count': frame_count,
        'points_data': points_data
    })


@app.route('/delete_objects/<name>', methods=['POST'])
def delete_objects(name):
    object_ids = request.json['object_ids']
    points_path = f'../data/points_json/{name}_points.json'
    areas_path = f'../data/areas_csv/{name}_SAM.csv'

    try:
        # Delete objects from points JSON
        with open(points_path, 'r') as f:
            points_data = json.load(f)

        for frame in points_data.values():
            for obj_id in object_ids:
                if obj_id in frame:
                    del frame[obj_id]

        with open(points_path, 'w') as f:
            json.dump(points_data, f)

        # Delete objects from areas CSV
        with open(areas_path, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)

        headers = rows[0]
        object_indices = [headers.index(obj_id) for obj_id in object_ids if obj_id in headers]

        for row in rows[1:]:
            for index in sorted(object_indices, reverse=True):
                del row[index]

        for index in sorted(object_indices, reverse=True):
            del headers[index]

        with open(areas_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows[1:])
        print("Done")
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error deleting objects: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/frame/<name>/<int:frame_number>')
def get_frame(name, frame_number):
    video_path = f'../inter/videos/{name}.mp4'
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if ret:
        _, buffer = cv2.imencode('.jpg', frame)
        return send_file(
            io.BytesIO(buffer),
            mimetype='image/jpeg',
            as_attachment=False
        )
    else:
        return "Frame not found", 404


@app.route('/save_points/<name>', methods=['POST'])
def save_points(name):
    points_data = request.json
    points_path = f'../data/points_json/{name}_points.json'
    areas_path = f'../data/areas_csv/{name}_SAM.csv'

    try:
        # Save points data to JSON file
        with open(points_path, 'w') as f:
            json.dump(points_data, f)

        # Calculate areas and update CSV file
        frame_areas = calculate_areas(points_data)
        update_areas_csv(areas_path, frame_areas)

        return jsonify({'success': True})
    except Exception as e:
        print(f"Error saving points: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


def calculate_areas(points_data):
    frame_areas = {}

    for frame_number, objects in points_data.items():
        frame_areas[frame_number] = {}

        for obj_id, data in objects.items():
            spline_points = data.get('spline_points', [])
            area = calculate_polygon_area(spline_points)
            frame_areas[frame_number][obj_id] = area

    return frame_areas


def update_areas_csv(filepath, frame_areas):
    if not os.path.exists(filepath):
        headers = [""] + list(frame_areas[next(iter(frame_areas))].keys())
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    with open(filepath, 'r') as f:
        reader = list(csv.reader(f))
        existing_data = {int(row[0]): row[1:] for row in reader[1:]}

    for frame_number, areas in frame_areas.items():
        existing_data[int(frame_number)] = [
            areas.get(f'obj_{i + 1}', '') for i in range(len(areas))
        ]

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([""] + [f'obj_{i + 1}' for i in range(len(existing_data[0]))])
        for frame_number, row in sorted(existing_data.items()):
            writer.writerow([frame_number] + row)


if __name__ == '__main__':
    app.run(debug=False, port=8080)


