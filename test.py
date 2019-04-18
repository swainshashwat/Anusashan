

# Init Video Capture
print('Init Video Capture')
cap = cv2.VideoCapture(0)

while True:

  # reading video
  ret, img = cap.read()

  # Display Image
  cv2.imshow("Facial Landmark detector", img)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    print('Quitting...')
    break

  
print('Releasing Resources...')
cap.release()
cv2.destroyAllWindows()    