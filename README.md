#Automatic Number Plate Recognition (ANPR) 🚘🔍

This project implements an Automatic Number Plate Recognition (ANPR) system using YOLOv8 and YOLOv11 for number plate detection, combined with EasyOCR for text recognition.
🧠 Overview

The ANPR system performs the following tasks:

    Detects vehicles and number plates using YOLO models (v8 & v11).

    Extracts the number plate region.

    Performs Optical Character Recognition (OCR) using EasyOCR to extract the alphanumeric characters from the plate.

    Saves results (text + image) to a specified output folder or text file.

📦 Features

    Real-time detection using webcam/video

    Image input support

    Dual YOLO model support for experimentation (YOLOv8 and YOLOv11)

    OCR integration using EasyOCR

    Results saving in text and image formats

🛠 Requirements

Install dependencies:

pip install ultralytics
pip install easyocr
pip install opencv-python

📁 Project Structure

anpr_project/
│
├── models/
│   ├── yolov8n.pt
│   └── yolov11.pt
│
├── input/
│   ├── car1.jpg
│   └── video.mp4
│
├── output/
│   ├── detected_images/
│   └── plate_numbers.txt
│
├── main.py
├── utils.py
└── README.md


📋 Output

    Bounding boxes on number plates

    Extracted text saved to plate_numbers.txt

    Annotated images saved to output/detected_images/

🧪 Models Used

    YOLOv8: Lightweight, fast, and accurate for number plate detection

    YOLOv11 (custom-trained): Tuned for enhanced accuracy in varied lighting and angles

    EasyOCR: Lightweight OCR for multilingual number plate recognition

📌 TODO

    Add GUI support (Tkinter/Streamlit)

    Fine-tune YOLOv11 on more diverse datasets

    Integrate with vehicle databases

🙌 Acknowledgements

    Ultralytics YOLOv8

    EasyOCR

    Custom datasets and preprocessing
