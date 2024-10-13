import face_recognition
import cv2
import numpy as np

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
jaime_image = face_recognition.load_image_file("C:/Users/sasha/Desktop/FR/references/jaime.jpg")
jaime_face_encoding = face_recognition.face_encodings(jaime_image)[0]

billy_image = face_recognition.load_image_file("C:/Users/sasha/Desktop/FR/references/billy.jpg")
billy_face_encoding = face_recognition.face_encodings(billy_image)[0]

van_image = face_recognition.load_image_file("C:/Users/sasha/Desktop/FR/references/van.jpg")
van_face_encoding = face_recognition.face_encodings(van_image)[0]

#-------------------------------------------------------------------------------------------

epif_image = face_recognition.load_image_file("C:/Users/sasha/Desktop/FR/references/epif.jpg")
epif_face_encoding = face_recognition.face_encodings(epif_image)[0]

pahom_image = face_recognition.load_image_file("C:/Users/sasha/Desktop/FR/references/pahom.jpg")
pahom_face_encoding = face_recognition.face_encodings(pahom_image)[0]

sanych_image = face_recognition.load_image_file("C:/Users/sasha/Desktop/FR/references/sanych.jpg")
sanych_face_encoding = face_recognition.face_encodings(sanych_image)[0]

#-------------------------------------------------------------------------------------------

# Create arrays of known face encodings and their names
known_face_encodings = [
    jaime_face_encoding,
    billy_face_encoding,
    van_face_encoding,
    epif_face_encoding,
    pahom_face_encoding,
    sanych_face_encoding
]

known_face_names = [
    "Jaime",
    "Billy",
    "Van",
    "Epifantsev",
    "Pahom",
    "Sanych"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face enqcodings in the frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        # If a match was found in known_face_encodings, just use the first one.
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()