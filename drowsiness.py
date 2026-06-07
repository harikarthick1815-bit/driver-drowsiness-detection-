import cv2
import time
from playsound import playsound 
import winsound

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

eye_closed_start = None
threshold = 2

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera not detected")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:

        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        face_gray = gray[y:y+h, x:x+w]
        face_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(face_gray)

        if len(eyes) == 0:

            if eye_closed_start is None:
                eye_closed_start = time.time()

            else:
                elapsed = time.time() - eye_closed_start

                if elapsed > threshold:
                    print("DROWSINESS DETECTED!")
                    winsound.PlaySound("alarm.wav",winsound.SND_FILENAME)
                    cv2.putText(frame,"DROWSY ALERT!",(50,50),
                                cv2.FONT_HERSHEY_SIMPLEX,1,
                                (0,0,255),3)
                    winsound.PlaySound("alarm.wav",winsound.SND_FILENAME)

        else:
            eye_closed_start = None

        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(face_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()