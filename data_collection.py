import cv2
import os

# Path for saving images (change this to 'B' folder or whichever sign you're capturing)
save_path = "C:/Users/rahul/OneDrive/Desktop/project1/colored_imgs/Z"
os.makedirs(save_path, exist_ok=True)

# Initialize the camera and hand detector
cap = cv2.VideoCapture(0)

# Using cvzone HandDetector for detecting hands
from cvzone.HandTrackingModule import HandDetector
hd = HandDetector(maxHands=2, detectionCon=0.8)  # Use two hands and a reasonable confidence level

# Set a limit on how many images to capture
image_limit = 100  # You can change this number to whatever you prefer
counter = 0  # Image counter

# Start data collection
print("Data collection started for B. Press 'q' to stop.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Mirror the image (flip horizontally)
    frame = cv2.flip(frame, 1)  # 1 flips the image horizontally

    # Detect hands in the frame
    hands, img = hd.findHands(frame,flipType=False)

    # Draw landmarks and bounding box (manual drawing)
    if hands:
        for hand in hands:
            # Get the bounding box for each hand
            x, y, w, h = hand['bbox']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw bounding box

            # Draw landmarks
            for lm in hand['lmList']:
                cv2.circle(frame, (lm[0], lm[1]), 5, (0, 255, 0), -1)  # Draw landmark points

    # Show the frame on the screen
    cv2.imshow("Frame", frame)

    # Store images when a hand is detected
    if hands and counter < image_limit:
        counter += 1
        filename = os.path.join(save_path, f"{counter}.jpg")
        cv2.imwrite(filename, frame)  # Save the mirrored frame as a color image
        print(f"Saved image: {filename}")

    # Stop if the limit is reached or if 'q' is pressed
    if counter >= image_limit or cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
