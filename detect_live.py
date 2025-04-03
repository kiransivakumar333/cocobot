import torch
import cv2
import numpy as np

# Load trained model
model = torch.hub.load("ultralytics/yolov5", "custom", path="coconut_maturity_model_new/coconut_maturity_model_new.pt")

# Define class names and their corresponding colors
CLASS_COLORS = {
    "Stage 1": (0, 0, 255),    # Red for not mature
    "Stage 2": (0, 255, 255),  # Yellow for partially mature
    "Stage 3": (0, 255, 0)     # Green for fully mature
}

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection
    results = model(frame)
    df = results.pandas().xyxy[0]  # Convert results to pandas DataFrame

    for index, row in df.iterrows():
        x1, y1, x2, y2 = int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"])
        confidence = row["confidence"]
        label = row["name"]  # Get the detected class name
        
        # Assign color based on class
        color = CLASS_COLORS.get(label, (255, 255, 255))  # Default to white if unknown

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Put label text
        label_text = f"{label} ({confidence:.2f})"
        cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show the frame
    cv2.imshow("YOLOv5 Live Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
