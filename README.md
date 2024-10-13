# Face Recognition Application

This project implements a real-time face recognition system using the `face_recognition` library and OpenCV.

## Features
- Real-time face detection via webcam
- Face recognition of known individuals
- Displays the name of recognized individuals on the video feed
- Unknown faces are labeled as "Unknown"
- Simple and efficient comparison using distance metrics

## Setup
1. Install dependencies:
   ```
   pip install face_recognition opencv-python numpy
   ```
2. Place reference images in the `references/` folder.
3. Update the paths in the script to match the location of your reference images.

## Running the Program
To start face recognition:
```
python face_recognition.py
```

Press 'q' to exit the program.

## Notes
- Ensure your webcam is accessible for video capture.
- You can add more faces by updating the `known_face_encodings` and `known_face_names` lists.

## Authors
This application is developed by Alexandr Kulakov
