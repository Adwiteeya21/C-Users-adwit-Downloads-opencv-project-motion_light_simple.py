import cv2
import time

# Start camera
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera. Please check your camera connection.")
    exit()

motion_detected = False
light_on = False
last_motion_time = time.time()
timeout = 5  # seconds to turn off light after no motion

# Background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=100, detectShadows=False)

print("Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (21, 21), 0)

    # Apply background subtraction to blurred grayscale image
    fgmask = fgbg.apply(blur)

    # Morphology to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False
    largest_contour = None
    largest_area = 0

    # Find the largest contour
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1500:
            motion_detected = True
            if area > largest_area:
                largest_area = area
                largest_contour = contour
    
    # Draw only one green rectangle around the largest moving object/person
    if largest_contour is not None:
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    current_time = time.time()

    if motion_detected:
        last_motion_time = current_time
        if not light_on:
            print("💡 Light ON (motion detected)")
            light_on = True
    elif light_on and (current_time - last_motion_time > timeout):
        print("💤 Light OFF (no motion)")
        light_on = False

    # Show video and motion detection
    cv2.imshow("Camera", frame)
    cv2.imshow("Motion Mask", fgmask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
