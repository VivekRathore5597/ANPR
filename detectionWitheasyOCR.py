import time
import cv2
from ultralytics import YOLO
import easyocr
import numpy as np
from sort.sort import Sort

# Initialize models
video_path = 'a.mp4'  # Path to your video
car_detection = YOLO('yolo11n.pt')  # Vehicle detection model
number_plate = YOLO('best.pt')  # License plate detection model

# Initialize EasyOCR and Sort tracker
reader = easyocr.Reader(['en'], gpu=True)
tracker = Sort()

vehicles = [2, 3, 5, 7]

# Open video feed
cam = cv2.VideoCapture(video_path)

frame_nu = -1
ret = True

# Helper function to associate license plates with cars
def get_car(license_plate, car_ids):
    x1, y1, x2, y2, score, cls_id = license_plate
    for car in car_ids:
        xcar1, ycar1, xcar2, ycar2, car_id = car
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            return car_id
    return -1

# Function to read the license plate text using OCR
def read_plate(cropped_plate):
    detections = reader.readtext(cropped_plate)
    for detection in detections:
        bbox, text, score = detection
        text = text.upper().replace(' ', '')  # Remove spaces and make text uppercase
        return text
    return None

# Segment characters using contours in the dilated edge image
def segment_characters(dilated_edges):
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    char_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 10 and h > 20:  # Filter out small contours
            char_boxes.append([x, y, x + w, y + h])
    
    # Sort character boxes from left to right
    char_boxes = sorted(char_boxes, key=lambda x: x[0])
    return char_boxes

# Process video frames
while ret:
    frame_nu += 1
    ret, frame = cam.read()
    if not ret:
        break
    dilated_edges = None

    frame = cv2.resize(frame, (1440, 720))  # Resize for consistency

    # Vehicle detection
    detections = car_detection(frame)[0]
    detection_boxes = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, id = detection
        if int(id) in vehicles:  # Check if detected object is a vehicle
            detection_boxes.append([x1, y1, x2, y2, score])

    if len(detection_boxes) > 0:
        track_ids = tracker.update(np.asarray(detection_boxes))  # Update vehicle tracking
    else:
        track_ids = []

    # License plate detection
    license_plate = number_plate(frame)[0]
    if license_plate:
        for plates in license_plate.boxes.data.tolist():
            x1, y1, x2, y2, score, id = plates
            car_id = get_car(plates, track_ids)  # Get associated car ID
            if car_id != -1:  # If license plate is associated with a car
                cropped_plate = frame[int(y1):int(y2), int(x1):int(x2), :]  # Crop the license plate region

                # Pre-process the cropped plate image for better OCR accuracy
                blurred_plate = cv2.GaussianBlur(cropped_plate, (7, 7), 1)
                gray_plate = cv2.cvtColor(blurred_plate, cv2.COLOR_BGR2GRAY)
                low_t = 30
                high_t = 100
                edges = cv2.Canny(gray_plate, low_t, high_t)

                # Find contours and sort them by area
                cnts, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]  # Keep the largest contours
                screenCnt = None
                for c in cnts:
                    perimeter = cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)
                    if len(approx) == 4:  # Find the rectangular contour
                        screenCnt = approx
                if screenCnt is not None:
                    cv2.drawContours(edges, [screenCnt], -1, (0, 255, 0), 3)  # Highlight detected plate contour

                # Apply morphological transformations (erosion and dilation)
                kernel = np.ones((1, 1), np.uint8)
                eroded = cv2.erode(edges, kernel, iterations=1)
                dilated_edges = cv2.dilate(eroded, kernel, iterations=1)

                # Segment characters from the dilated edges
                char_boxes = segment_characters(dilated_edges)
                for (x1_char, y1_char, x2_char, y2_char) in char_boxes:
                    cv2.rectangle(cropped_plate, (x1_char, y1_char), (x2_char, y2_char), (0, 255, 0), 2)

                # Perform OCR to read the license plate text
                license_plate_text = read_plate(dilated_edges)
                if license_plate_text:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    cv2.putText(frame, license_plate_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Draw bounding boxes for detected vehicles
    for detection in detection_boxes:
        x1, y1, x2, y2, score = detection
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(frame, f"Vehicle {score:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Show the processed video frame
    cv2.imshow('Video with Detections', frame)
    if dilated_edges is not None:
        cv2.imshow("Dilated Edges", dilated_edges)  # Show dilated edges for debugging
    #cv2.imshow("Segmented Plate", cropped_plate)  # Optionally, display the segmented license plate

    # Exit on 'q' key press
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up and release video capture
cv2.destroyAllWindows()
cam.release()
