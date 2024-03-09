import cv2
# Initialize video capture object
cap = cv2.VideoCapture(0)
# Define previous frame
prev_frame = None
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and improve accuracy of motion detection
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # If there is no previous frame, initialize it
    if prev_frame is None:
        prev_frame = gray
        continue

    # Calculate absolute difference between current frame and previous frame
    frame_diff = cv2.absdiff(prev_frame, gray)

    # Apply a threshold to identify regions with significant differences
    thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)[1]

    # Dilate the thresholded image to fill in holes
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours of the thresholded image
    contours, _ = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Loop over the contours
    for contour in contours:
        # If the contour area is small, ignore it
        if cv2.contourArea(contour) < 1000:
            continue

        # Draw bounding box around the contour
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Motion Detection", frame)

    # Set current frame as previous frame for next iteration
    prev_frame = gray

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release video capture object and close all windows
cap.release()
cv2.destroyAllWindows()