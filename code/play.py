from ultralytics.models.fastsam import FastSAMPredictor
import time
# Create FastSAMPredictor
overrides = dict(conf=0.25, task="segment", mode="predict", model="weights/FastSAM-s.pt", save=False, imgsz=800,
                 show=True)
predictor = FastSAMPredictor(overrides=overrides)

# Segment everything
everything_results = predictor("/Users/ypm/Desktop/Track/code/2/DoS_4MDa_100ppm_BrijO10_0.1CMC_vial5_28Jul (3)/00100.jpg")

# Prompt inference
point_results = predictor.prompt(everything_results, points=[500, 500])

from ultralytics import SAM

# Load a model
model = SAM("sam_b.pt")

# Display model information (optional)
model.info()

# Run inference with points prompt
results = model("ultralytics/assets/zidane.jpg", points=[500, 500], labels=[1])

time.sleep(15)
