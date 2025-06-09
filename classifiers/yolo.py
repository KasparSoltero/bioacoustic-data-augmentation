from ultralytics import YOLO

# Load a model
# model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11n-seg.pt")  # load an official model

# Run batched inference on a list of images
# results = model(["tests/image1.png", "tests/image2.png"])  # return a list of Results objects
# results = model(["tests/image5.jpg"])

# # Process results list
# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     # xy = result.masks.xy  # mask in polygon format
#     # xyn = result.masks.xyn  # normalized
#     # masks = result.masks.data  # mask in matrix format (num_objects x H x W)
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     obb = result.obb  # Oriented boxes object for OBB outputs
#     result.show()  # display to screen
#     result.save(filename="result.jpg")  # save to disk

# Train the model with MPS
results = model.train(data="augmented_dataset_output/artificial_dataset/dataset.yaml", epochs=10, imgsz=640, device="mps")