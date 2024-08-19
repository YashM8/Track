from networks import fast_sam, vid_sam
from utilities import process_gray_resize_to_video, plot_csv_data
import gc
from app import app


class Tracker:
    def __init__(self, name: str, scale_down: int = 3, skip_images: int = 3, suppress: float = 0.1,
                 conf_prob: float = 0.4, img_h: int = 692, img_w: int = 1029, debug_js: bool = False,
                 device: str = "cpu", stream_from_mem: bool = True, dp_threshold: int = 12):
        self.name = name
        self.scale_down = scale_down
        self.skip_images = skip_images
        self.suppress = suppress
        self.conf_prob = conf_prob
        self.img_h = img_h
        self.img_w = img_w
        self.debug_js = debug_js
        self.device = device
        self.stream = stream_from_mem
        self.dp_threshold = dp_threshold

        print(f"\n{'-' * 70}\nDevice - {device}. THIS SHOULD BE 'CUDA' OR 'GPU' or 'TPU or 'MPS''\n{'-' * 70}\n")

    def preprocess(self):
        print(f"\n{'-' * 40}\nTurning into video\n{'-' * 40}\n")
        process_gray_resize_to_video(f"../data/raw_data/{self.name}",
                                     f"../inter/videos/{self.name}.mp4",
                                     skip=3, scale_down=3)
        gc.collect()

    def predict(self):
        self._detect()
        self._segment()

    def _detect(self):
        print(f"\n{'-' * 40}\nGetting bounding boxes and tracking\n{'-' * 40}\n")
        fast_sam(self.name, iou=self.suppress, conf=self.conf_prob,
                 image_h=self.img_h, image_w=self.img_w, stream=self.stream)
        gc.collect()

    def _segment(self):
        print(f"\n{'-' * 40}\nSegmenting and collecting areas\n{'-' * 40}\n")
        vid_sam(self.name, self.device, dp_threshold=self.dp_threshold)
        gc.collect()

        print(f"\n{'-' * 40}\nInference Done\n{'-' * 40}\n")

    def plot_areas(self, show=False):
        print(f"\n{'-' * 40}\nPlotting data\n{'-' * 40}\n")
        plot_csv_data(self.name, show=show)

    def open_manual(self):
        app.run(debug=self.debug_js, port=8080)

    def __str__(self):
        return (f"Name: {self.name}\n"
                f"Scale Down: {self.scale_down}\n"
                f"Skip Images: {self.skip_images}\n"
                f"Suppress: {self.suppress}\n"
                f"Confidence Probability: {self.conf_prob}\n"
                f"Image Height: {self.img_h}\n"
                f"Image Width: {self.img_w}\n"
                f"Debug JS: {self.debug_js}\n"
                f"Device: {self.device}\n"
                f"Stream from Memory: {self.stream}")
