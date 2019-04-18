import dlib
import cv2
import numpy as np
from renderFace import renderFace

# Landmark model location
PREDICTOR_PATH = "landmark_model/shape_predictor_68_face_landmarks.dat"

# Get the face detector
print('Load Face Detector')
faceDetector = dlib.get_frontal_face_detector()

# The landmark detector is implemented in the shape_predictor class
print('Load Facial Landmark Predictor')
landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)

# Init Video Capture
print('Init Video Capture')
cap = cv2.VideoCapture(0)

while True:

  # reading video
  ret, img = cap.read()

  # Detect faces in the image
  faceRects = faceDetector(img, 0)
  
  # Loop over all detected face rectangles
  for i in range(0, len(faceRects)):
    
    newRect = dlib.rectangle(int(faceRects[i].left()),int(faceRects[i].top()),
                            int(faceRects[i].right()),int(faceRects[i].bottom()))

    # For every face rectangle, run landmarkDetector
    landmarks = landmarkDetector(img, newRect)


    # Draw landmarks on face
    renderFace(img, landmarks)

  # Display Image
  cv2.imshow("Facial Landmark detector", img)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    print('Quitting...')
    break

  
print('Releasing Resources...')
cap.release()
cv2.destroyAllWindows()