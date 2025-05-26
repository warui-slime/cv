import cv2
import numpy as np

# Load image
imagePath = "orignal.png"
image = cv2.imread(imagePath)


# Convert image to RGB for consistent handling if needed (OpenCV loads BGR)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Parameters
Location = [120, 190]  # y, x
PatchSize = [128, 64]  # height, width
W = 8  # region size inside patch

# Draw patch rectangle with grid lines and save image
np_im = image.copy()

cv2.rectangle(np_im,
              (Location[1], Location[0]),
              (Location[1] + PatchSize[1], Location[0] + PatchSize[0]),
              (0, 0, 255), 1)

numlinesY = PatchSize[0] // 8
numlinesX = PatchSize[1] // 8

for x in range(numlinesX):
    cv2.line(np_im,
             (Location[1] + 8 * (x + 1), Location[0]),
             (Location[1] + 8 * (x + 1), Location[0] + PatchSize[0]),
             (0, 0, 255), 1)

for y in range(numlinesY):
    cv2.line(np_im,
             (Location[1], Location[0] + 8 * (y + 1)),
             (Location[1] + PatchSize[1], Location[0] + 8 * (y + 1)),
             (0, 0, 255), 1)

cv2.imwrite("patch_with_grid.png", np_im)

# Calculate gradients with Sobel (per channel)
gx = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 1, 0, ksize=1)
gy = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 0, 1, ksize=1)

# Calculate magnitude and angle per channel
mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

# Convert to int for simplicity
mag = mag.astype(int)
angle = angle.astype(int)

# Find max gradient channel per pixel
maxChan = np.argmax(mag, axis=2)

# Extract max gradient magnitude for each pixel
maxmag = np.zeros(maxChan.shape, dtype=int)
rows, cols = maxChan.shape
for r in range(rows):
    for c in range(cols):
        maxmag[r, c] = mag[r, c, maxChan[r, c]]

# Extract max angle corresponding to max gradient channel
maxangle = np.zeros_like(maxmag, dtype=int)
for r in range(rows):
    for c in range(cols):
        maxangle[r, c] = angle[r, c, maxChan[r, c]]

# Save max magnitude image normalized to 0-255 grayscale
maxmag_img = (maxmag / maxmag.max() * 255).astype(np.uint8)
cv2.imwrite("max_gradient_magnitude.png", maxmag_img)

# Function to map angles to [0,180)
def anglemapper(x):
    return x - 180 if x >= 180 else x

vfunc = np.vectorize(anglemapper)
mappedAngles = vfunc(maxangle)

# Histogram creation function
def createHist(AngArray, MagArray, BS=20, BINS=9):
    hist = np.zeros(BINS, dtype=float)
    for r in range(AngArray.shape[0]):
        for c in range(AngArray.shape[1]):
            angle_val = AngArray[r, c]
            mag_val = MagArray[r, c]
            binel, rem = divmod(angle_val, BS)
            weightR = rem / BS
            weightL = 1 - weightR
            binL = int(binel)
            binR = (binL + 1) % BINS
            hist[binL] += mag_val * weightL
            hist[binR] += mag_val * weightR
    return hist

# Calculate histograms for the 4 subregions of the patch
histList = []
for dy in [0, W]:
    for dx in [0, W]:
        sub_angles = mappedAngles[Location[0]+dy:Location[0]+dy+W, Location[1]+dx:Location[1]+dx+W]
        sub_mag = maxmag[Location[0]+dy:Location[0]+dy+W, Location[1]+dx:Location[1]+dx+W]
        hist = createHist(sub_angles, sub_mag)
        histList.append(hist)

# Concatenate histograms into one vector
histRegion = np.concatenate(histList)

# L2 normalize
epsilon = 1e-6
l2norm = np.linalg.norm(histRegion)
histRegionNormed = histRegion / (l2norm + epsilon)

# Print normalized histogram vector
print("Normalized histogram vector (36 bins):")
print(histRegionNormed)

# Save the patch image itself for reference (with RGB colors)
patch = image_rgb[Location[0]:Location[0]+PatchSize[0], Location[1]:Location[1]+PatchSize[1]]
patch_bgr = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)  # convert back for saving
cv2.imwrite("patch.png", patch_bgr)
