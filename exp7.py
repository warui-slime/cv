import cv2
import numpy as np
import glob
import os
import yaml

# Criteria for corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare 7x7 chessboard object points (0,0,0), (1,0,0), ..., (6,6,0)
objp = np.zeros((49, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2)

# Lists for 3D and 2D points
objpoints = []
imgpoints = []

# Load images
image_paths = glob.glob('/content/dd/*.jpg')
print(f"Found {len(image_paths)} images.")

# Output directory
output_dir = 'results'
os.makedirs(output_dir, exist_ok=True)

valid_count = 0
for i, path in enumerate(image_paths):
    img = cv2.imread(path)
    if img is None:
        print(f"Could not load {path}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)

    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and save image with corners
        img_with_corners = cv2.drawChessboardCorners(img, (7, 7), corners2, ret)
        out_path = os.path.join(output_dir, f'calib_result_{valid_count+1}.jpg')
        cv2.imwrite(out_path, img_with_corners)
        print(f"Saved detected corners to {out_path}")
        valid_count += 1
    else:
        print(f"Chessboard not found in {path}")

print(f"Valid images used for calibration: {valid_count}")
if valid_count == 0:
    print("No valid images found. Exiting.")
    exit()

# Calibrate camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Output calibration results
print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist)

# Save calibration data to YAML
calibration_data = {
    'camera_matrix': mtx.tolist(),
    'dist_coeff': dist.tolist()
}
with open('calibration_matrix.yaml', 'w') as f:
    yaml.dump(calibration_data, f)

print("Calibration data saved to calibration_matrix.yaml")
