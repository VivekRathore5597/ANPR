from ultralytics import YOLO
import cv2
import pytesseract
from datetime import datetime

# Load the model
model = YOLO("best.pt")

# Configure the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'E:/tersseract/tesseract.exe'

# Open the video file
cam = cv2.VideoCapture("a.mp4")
list= []
prossed = set()
my_file = 'detected_text.txt'


with open(my_file, 'w') as f:
    while cam.isOpened():
        ret, frame = cam.read()
        if not ret:
            break
       
        
        # Perform detection
        results = model(frame)
        
        # Draw bounding boxes and labels on the frame
        if results:
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    conf = box.conf[0]
                    cls = box.cls[0]
                    label = f'{model.names[int(cls)]}: {conf*100:.2f}'
                    # Draw the bounding box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Crop the region of interest (ROI) for OCR
                    roi = frame[int(y1):int(y2), int(x1):int(x2)]
                    
                    # Perform OCR on the ROI
                    text = pytesseract.image_to_string(roi, config='--psm 6').strip()
                    text= text.replace("("," ").replace(")"," ").replace("["," ").replace("]"," ").replace(","," ").replace("."," ")
                    if text:
                    
                        cv2.putText(frame, text ,(int(x1),int(y2)+25), cv2.FONT_HERSHEY_SIMPLEX ,1,(255,0,0),2)
                    if text not in prossed:
                        prossed.add(text)
                        list.append(text)
                        

                    cv2.rectangle(frame, (int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),1)  
                              
                   

        # Display the frame
        cv2.imshow("video",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cam.release()
cv2.destroyAllWindows()

