import cv2

stream = cv2.VideoCapture(0)

if not stream.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = stream.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break


    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

stream.release()
cv2.destroyAllWindows()