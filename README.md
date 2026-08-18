# Sturdy-Octo-Disco-Adding-Sunglasses-for-a-Cool-New-Look

Sturdy Octo Disco is a fun project that adds sunglasses to photos using image processing.

Welcome to Sturdy Octo Disco, a fun and creative project designed to overlay sunglasses on individual passport photos! This repository demonstrates how to use image processing techniques to create a playful transformation, making ordinary photos look extraordinary. Whether you're a beginner exploring computer vision or just looking for a quirky project to try, this is for you!

## Features:
- Detects the face in an image.
- Places a stylish sunglass overlay perfectly on the face.
- Works seamlessly with individual passport-size photos.
- Customizable for different sunglasses styles or photo types.

## Technologies Used:
- Python
- OpenCV for image processing
- Numpy for array manipulations

## How to Use:
1. Clone this repository.
2. Add your passport-sized photo to the `images` folder.
3. Run the script to see your "cool" transformation!

## Applications:
- Learning basic image processing techniques.
- Adding flair to your photos for fun.
- Practicing computer vision workflows.

Feel free to fork, contribute, or customize this project for your creative needs!

## Program:

```
import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Load face image
# -----------------------------
faceImage = cv2.imread("image.jpeg")

if faceImage is None:
    raise ValueError("Could not load image.jpeg")

# Convert to grayscale
gray = cv2.cvtColor(faceImage, cv2.COLOR_BGR2GRAY)

# -----------------------------
# Detect face
# -----------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(100, 100)
)

if len(faces) == 0:
    raise ValueError("No face detected.")

# Take the largest face
x_face, y_face, w_face, h_face = max(
    faces, key=lambda f: f[2] * f[3]
)

# -----------------------------
# Detect eyes inside face
# -----------------------------
face_gray = gray[
    y_face:y_face+h_face,
    x_face:x_face+w_face
]

eyes = eye_cascade.detectMultiScale(
    face_gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)

if len(eyes) < 2:
    raise ValueError("Could not detect both eyes.")

# Sort eyes from left to right
eyes = sorted(eyes, key=lambda e: e[0])

# Take first two eyes
eye1 = eyes[0]
eye2 = eyes[1]

# Convert eye coordinates to original image coordinates
ex1, ey1, ew1, eh1 = eye1
ex2, ey2, ew2, eh2 = eye2

ex1 += x_face
ey1 += y_face

ex2 += x_face
ey2 += y_face

# -----------------------------
# Calculate glasses position
# -----------------------------

# Center of both eyes
eye_center_x = (
    (ex1 + ew1 / 2) +
    (ex2 + ew2 / 2)
) / 2

eye_center_y = (
    (ey1 + eh1 / 2) +
    (ey2 + eh2 / 2)
) / 2

# Distance between eyes
eye_distance = abs(
    (ex2 + ew2 / 2) -
    (ex1 + ew1 / 2)
)

# -----------------------------
# Load sunglasses
# -----------------------------
glassPNG = cv2.imread(
    "sunglass.png",
    cv2.IMREAD_UNCHANGED
)

if glassPNG is None:
    raise ValueError("Could not load sunglass.png")

# If no alpha channel
if glassPNG.shape[2] == 3:

    glassPNG = cv2.cvtColor(
        glassPNG,
        cv2.COLOR_BGR2BGRA
    )

    # White pixels become transparent
    white_mask = np.all(
        glassPNG[:, :, :3] >= 240,
        axis=2
    )

    glassPNG[white_mask, 3] = 0

# -----------------------------
# Resize sunglasses
# -----------------------------

original_h, original_w = glassPNG.shape[:2]

# Target glasses width
target_width = int(eye_distance * 2.5)

scale = target_width / original_w

new_w = int(original_w * scale)
new_h = int(original_h * scale)

glassPNG = cv2.resize(
    glassPNG,
    (new_w, new_h),
    interpolation=cv2.INTER_AREA
)

h, w = glassPNG.shape[:2]

# -----------------------------
# Position sunglasses
# -----------------------------

# Center sunglasses on eyes
x = int(eye_center_x - w / 2)

# Slightly above eye center
y = int(eye_center_y - h * 0.45)

# -----------------------------
# Boundary check
# -----------------------------

x1 = max(0, x)
y1 = max(0, y)

x2 = min(faceImage.shape[1], x + w)
y2 = min(faceImage.shape[0], y + h)

# Corresponding sunglasses coordinates
gx1 = x1 - x
gy1 = y1 - y
gx2 = gx1 + (x2 - x1)
gy2 = gy1 + (y2 - y1)

# -----------------------------
# Alpha blending
# -----------------------------

glass_region = glassPNG[
    gy1:gy2,
    gx1:gx2
]

glassBGR = glass_region[:, :, :3]

alpha = glass_region[:, :, 3] / 255.0

alpha = alpha[:, :, np.newaxis]

face_region = faceImage[
    y1:y2,
    x1:x2
]

# Blend
blended = (
    face_region.astype(float) * (1 - alpha)
    +
    glassBGR.astype(float) * alpha
)

blended = np.uint8(blended)

# Put back into image
result = faceImage.copy()

result[
    y1:y2,
    x1:x2
] = blended

# -----------------------------
# Display
# -----------------------------

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(faceImage, cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title("Sunglasses Automatically Positioned")
plt.axis("off")

plt.show()
```

## Output:
<img width="1078" height="649" alt="image" src="https://github.com/user-attachments/assets/da769542-6a22-4bdd-90fd-09add0e47898" />
