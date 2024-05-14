from ultralytics import YOLO
from PIL import Image


# Create a new YOLO model from scratch
model = YOLO('yolov8n.yaml')

# Load a pretrained YOLO model (recommended for training)
model = YOLO('runs/detect/train/weights/best.pt')
im1 = Image.open("images/sample2.jpg")

results = model.predict(source=im1, save=True)
print(results)
# for box in result.boxes:
#   class_id = result.names[box.cls[0].item()]
#   cords = box.xyxy[0].tolist()
#   cords = [round(x) for x in cords]
#   conf = round(box.conf[0].item(), 2)
#   print("Object type:", class_id)
#   print("Coordinates:", cords)
#   print("Probability:", conf)
#   print("---")