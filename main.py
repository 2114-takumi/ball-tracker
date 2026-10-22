import cv2
import time

cap = cv2.VideoCapture(0)

start_time= time.perf_counter()
frame_count = 0


while True:
    ret, frame = cap.read()
    if ret:
        frame_count += 1
        cv2.imshow("Camera", frame)
        key = cv2.waitKey(1)
        current_time = time.perf_counter()
        elapsed_time = current_time - start_time
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            print(f"FPS: {fps:.2f}")
            start_time = current_time
            frame_count = 0   
        if key & 0xFF == ord('q'):
            break

    if not ret:
        break
cv2.destroyAllWindows()
cap.release()