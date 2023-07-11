import cv2

# Load the Haar cascade file for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the Haar cascade file for smile detection
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# Start the video capture
video_capture = cv2.VideoCapture(0)

while True:
    # Read the current frame from the video stream
    ret, frame = video_capture.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate over each detected face
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Extract the region of interest (ROI) within the face rectangle
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect smiles within the ROI
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=20, minSize=(30, 30))

        # Check if a smile is detected
        if len(smiles) > 0:
            # Capture the selfie by saving the current frame
            cv2.imwrite("selfie.jpg", frame)

            # Display a message
            cv2.putText(frame, 'Smile detected! Selfie captured.', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
    # Display the frame
    cv2.imshow('Smile Detection', frame)

    # Check for the 'q' key press to exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
video_capture.release()
cv2.destroyAllWindows()
