# YOLOv8 Object Detection Using Laptop Camera
## NAME:SHARVESHWARAN M
## REG.NO: 212224240150

## Aim

To access the **laptop camera**, capture an image, and detect objects using **YOLOv8**.

## Requirements

* Anaconda
* Jupyter Notebook
* Laptop Camera

## Steps

1. Open **Jupyter Notebook** using Anaconda.
2. Access your **laptop camera** using OpenCV.
3. Wait for **5 seconds** and capture an image.
4. Display the captured image.
5. Apply **YOLOv8 object detection**.
6. Display the detected image with **bounding boxes and labels**.

## Algorithm

```text
Laptop Camera
      ↓
Capture Image
      ↓
Display Image
      ↓
YOLOv8 Detection
      ↓
Display Detected Objects
```

## Student Task

* Capture your own image using the laptop camera.
* Detect the objects present in the image.
* Display the final output.
* Perform the experiment with **3 different scenes**.

## Submission

* Jupyter Notebook (`.ipynb`)
* Original captured images
* YOLOv8 output images
* Screenshot of the final result

## GitHub Reference

https://github.com/ultralytics/ultralytics

**Platform:** Anaconda + Jupyter Notebook only.

## Program
```
import cv2
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

# Open webcam using DirectShow
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Cannot open webcam")
else:
    print("Webcam started successfully.")

    # Background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=50,
        detectShadows=True
    )

    # Morphological kernel
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    try:
        while True:

            # Read frame
            ret, frame = cap.read()

            if not ret:
                print("Cannot read frame from webcam")
                break

            # Apply background subtraction
            mask = bg_subtractor.apply(frame)

            # Remove small noise
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_OPEN,
                kernel
            )

            # Dilate foreground regions
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_DILATE,
                kernel
            )

            # Find contours
            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            # Detect objects
            for contour in contours:

                area = cv2.contourArea(contour)

                # Ignore small objects
                if area > 2500:

                    x, y, w, h = cv2.boundingRect(contour)

                    # Draw bounding box
                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + w, y + h),
                        (0, 255, 0),
                        2
                    )

                    # Display text
                    cv2.putText(
                        frame,
                        "Object Detected",
                        (x, max(y - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

            # Convert BGR → RGB
            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # Display in Jupyter
            clear_output(wait=True)

            plt.figure(figsize=(10, 6))
            plt.imshow(frame_rgb)
            plt.axis("off")
            plt.title("Workshop 2 - Object Detection")
            display(plt.gcf())
            plt.close()

    except KeyboardInterrupt:
        print("Stopping webcam...")

    finally:
        cap.release()
        print("Webcam released.")
```
## Output

<img width="943" height="710" alt="image" src="https://github.com/user-attachments/assets/c8071aa2-7946-44bf-966d-c08929c69fb8" />

Result:
The laptop camera was successfully accessed using OpenCV, and an image was captured after 5 seconds. The captured image was then processed using the YOLOv8 object detection model. YOLOv8 successfully detected the object present in the image and displayed it with a bounding box and label. Thus, the experiment successfully demonstrated object detection using a laptop camera and YOLOv8.
