import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera not accessible")
    exit()

print("Camera started successfully! Press Q to exit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame capture failed")
        break

    cv2.imshow("Unknown Face Recognition - Live Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
