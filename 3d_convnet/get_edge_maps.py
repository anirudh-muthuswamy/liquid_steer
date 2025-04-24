import os
import cv2
import numpy as np

#get canny binary edge maps (with parts of the image above the road) cropped out
#this is used as input to the 3d convnet

def process_images(input_dir, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    exts = ('.jpg', '.jpeg')

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(exts):
            continue

        img_path = os.path.join(input_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"(couldnt read) {fname}")
            continue

        height, width = img.shape[:2]

        # grayscale + blur
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)

        # canny edges
        edges = cv2.Canny(blur, 50, 70)

        # build and apply ROI mask
        mask = np.zeros_like(edges)
        roi_corners = np.array([[
            (0, height),
            (width, height),
            (width, int(height * 0.52)),
            (0, int(height * 0.52))]], dtype=np.int32)
        cv2.fillPoly(mask, roi_corners, 255)
        roi_edges = cv2.bitwise_and(edges, mask)

        y0 = int(height*0.52)
        cropped_edges = roi_edges[y0:,:]

        # save masked edges with the same filename
        out_path = os.path.join(output_dir, fname)
        cv2.imwrite(out_path, cropped_edges)
        print(f"Saved ROI edges to {out_path}")

if __name__ == "__main__":
    input_folder  = "data/sullychen/07012018/data"
    output_folder = "data/sullychen/07012018/edge_maps"
    process_images(input_folder, output_folder)
