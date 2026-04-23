import cv2
import numpy as np
import matplotlib.pyplot as plt

def save_debug_image(image, title="Image", filename="output.png", cmap='gray'):
    plt.figure()
    plt.title(title)
    plt.imshow(image if cmap != 'gray' else image, cmap=cmap)
    plt.axis('off')
    plt.savefig(filename)
    plt.close()

debug_capture = False

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("can't receive frame")
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([90, 100, 50])
    upper_blue = np.array([135, 255, 255])

    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = np.ones((5, 5), np.uint8)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = frame.copy()
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

    # Output list of contours
    contour_list = [cnt.reshape(-1, 2).tolist() for cnt in contours]
    print("Contours:", contour_list)

    cv2.imshow("blue", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
