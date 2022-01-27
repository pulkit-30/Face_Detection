import cv2 as cv

# Reading a video
capture = cv.VideoCapture('../videos/sample.mov')

# reading haarcascade_frontalface_default.xml
haar = cv.CascadeClassifier('../haarcascade_frontalface_default.xml')


# displaying video and detecting face frame_by_frame
while True:
    # reading each frame of a video
    isTrue, frame = capture.read()
    # coverting rgb image to a gray img
    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    # detecting face using haar cascadeClassifier
    face_rect = haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    # printing number of faces recognized
    print("number of faces detected --> ", len(face_rect))
    # drawing a rectangle over the detected face in the frame
    for x, y, w, h in face_rect:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
    # displaying the frame
    cv.imshow('Video', frame)
    # terminating condition
    if cv.waitKey(20) & 0xFF == ord('d'):
        break


# release pointer once done
capture.release()
# finally, destroying all the windows
cv.destroyAllWindows()
