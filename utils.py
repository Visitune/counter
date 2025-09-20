from PIL import Image, ImageDraw
import numpy as np
import cv2

def pil_to_cv(img_pil):
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def cv_to_pil(img_cv):
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

def draw_rois(image_pil, rois, color=(0,255,0), width=3):
    img = image_pil.copy()
    d = ImageDraw.Draw(img)
    for x1,y1,x2,y2 in rois:
        d.rectangle([x1,y1,x2,y2], outline=(color[0],color[1],color[2]), width=width)
    return img

def overlay_points(image_pil, points, color=(0,255,0), radius=4):
    img = pil_to_cv(image_pil.copy())
    for (x,y) in points:
        cv2.circle(img, (int(x),int(y)), radius, color, thickness=-1, lineType=cv2.LINE_AA)
    return cv_to_pil(img)

def make_heatmap_from_points(image_pil, points, sigma=12, alpha_img=0.6, alpha_hm=0.6):
    h, w = image_pil.height, image_pil.width
    hm = np.zeros((h, w), dtype=np.float32)
    for (x, y) in points:
        if 0 <= x < w and 0 <= y < h:
            hm[int(y), int(x)] += 1.0
    hm = cv2.GaussianBlur(hm, (0,0), sigma)
    if hm.max() > 0:
        hm = hm / hm.max()
    hm_color = cv2.applyColorMap((hm*255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(pil_to_cv(image_pil), alpha_img, hm_color, alpha_hm, 0)
    return cv_to_pil(overlay)

def apply_rois_mask(img_cv, rois):
    if not rois:
        return img_cv
    mask = np.zeros(img_cv.shape[:2], dtype=np.uint8)
    for x1,y1,x2,y2 in rois:
        cv2.rectangle(mask, (x1,y1), (x2,y2), 255, -1)
    return cv2.bitwise_and(img_cv, img_cv, mask=mask)

def adaptive_preprocess(img_cv, invert=True):
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 5)
    if invert and bw.mean() > 127:
        bw = 255 - bw
    bw = cv2.medianBlur(bw, 3)
    return bw

def morph_cleanup(bw, open_ksize=3, close_ksize=3):
    if open_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k)
    if close_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k)
    return bw

def count_connected_components(bw, min_area=25, max_area=5000):
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)
    points = []
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            cx, cy = centroids[i]
            points.append((float(cx), float(cy)))
    return points
