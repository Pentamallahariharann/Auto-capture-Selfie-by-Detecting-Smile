import cv2
import datetime
import dlib

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def is_smiling(shape):
    # Calculate the mouth aspect ratio to determine if the person is smiling
    # Mouth aspect ratio = (distance between top lip and bottom lip) / (distance between left corner and right corner of the mouth)
    left_corner = shape.part(48).x
    right_corner = shape.part(54).x
    top_lip = (shape.part(50).y + shape.part(51).y) / 2
    bottom_lip = (shape.part(58).y + shape.part(59).y) / 2

    mouth_width = right_corner - left_corner
    mouth_height = bottom_lip - top_lip
    if mouth_width <= 0:
        return False
    mouth_aspect_ratio = mouth_height / mouth_width

    # A smiling mouth has a higher aspect ratio, adjust the threshold based on your use case
    return mouth_aspect_ratio > 0.3  # Adjust this threshold value to your liking (0.3 indicates a simple smile)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        shape = predictor(gray, face)

        if is_smiling(shape):
            cv2.putText(frame, "Smiling", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Save the selfie when smiling
            time_stamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            file_name = f'selfie-{time_stamp}.png'
            cv2.imwrite(file_name, frame)

    cv2.imshow('cam star', frame)

    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
