from ultralytics import YOLO


# Create a new YOLO model from scratch
model = YOLO('yolov8n.yaml')

# Load a pretrained YOLO model (recommended for training)
model = YOLO('runs/detect/train/weights/best.pt')


dataset_path = 'datasets/chicken'

model.dataset=dataset_path
results = model.train(data='yolov8_config.yaml', epochs=200)
