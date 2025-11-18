# Canny-Edge-Detection-Dipt_ws_3

## Experiment: Detecting Edges using the Canny Algorithm

### Aim

Implement the Canny edge detection algorithm on a sample image, produce edge maps for different parameter settings, and analyze how Gaussian smoothing and the low/high thresholds affect the detected edges.

### Apparatus

* Computer with Python 3.8+
* Python packages: opencv-python, numpy, matplotlib
  Install with: pip install opencv-python numpy matplotlib
* Input image file: sample.jpg (place it in project folder)
* Text editor / IDE or Jupyter Notebook

### Theory (brief)

Canny edge detection uses a pipeline of steps to produce clean edge maps:

1. *Gaussian smoothing* — reduces image noise (controlled by kernel size and sigma).
2. *Gradient calculation* — Sobel operators compute gradient magnitude and direction.
3. *Non-maximum suppression* — thins edges by keeping local maxima along the gradient direction.
4. *Double thresholding* — pixels are classified as strong, weak or non-edge using two thresholds (low and high).
5. *Edge tracking by hysteresis* — weak pixels connected to strong pixels are preserved; others are suppressed.

Important parameters:

* Gaussian kernel size (odd integer) and sigma → denoising strength.
* threshold1 (low) and threshold2 (high) → determine sensitivity. Typical heuristic: threshold2 ≈ 2–3 × threshold1.

---

### Procedure

1. Load sample.jpg in grayscale.
2. Apply Gaussian blur with multiple (kernel_size, sigma) settings.
3. Run Canny with different (low, high) threshold pairs.
4. Visualize original, blurred, and edge images side-by-side to compare.
5. Save representative edge outputs.

---

### Code (Python)

Save as canny_experiments.py. Place sample.jpg in the same folder and run the script.

python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('profile.jpg')  # Replace with your image path
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Apply Canny edge detector
canny_edges = cv2.Canny(gray_image, 50, 150)

# Display results using Matplotlib
plt.figure(figsize=(12, 12))

# Original Image
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')



# Canny Edge Detection
plt.subplot(2, 2, 4)
plt.imshow(canny_edges, cmap='gray')
plt.title('Canny Edge Detection')
plt.axis('off')

# Show all results
plt.tight_layout()
plt.show()



---

### Output (what you will see)


---

### Result

Canny edge detection is effective for extracting clean contours when parameters are tuned to the specific image characteristics. There is an inherent trade-off: smoothing and high thresholds yield cleaner but sparser edges; less smoothing and lower thresholds yield richer edges with more noise.
