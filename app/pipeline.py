
import os
import io
from typing import Any, Dict
from PIL import Image
import numpy as np
import os
USE_REMOTE = os.environ.get("USE_REMOTE_OCR", "1") == "1"  
try:
    from .remote_ocr import call_remote_ocr
except Exception:
    try:
        from app.remote_ocr import call_remote_ocr
    except Exception:
        call_remote_ocr = None


def _is_dummy(dummy_flag):
    if dummy_flag is not None:
        return bool(dummy_flag)
    return os.environ.get("DUMMY_MODE", "1") == "1"

class ModelLoadError(RuntimeError):
    pass

# ---------------------------
# Stage 1: Orientation, deskew, crop, upsample (uses your preprocess_image)
# ---------------------------
def correct_orientation_advanced_cv(image: np.ndarray) -> np.ndarray:
    import cv2, numpy as np
    # (copied logic from your code, unchanged aside from local imports)
    confidences = []
    angles = [0, 90, 180, 270]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for angle in angles:
        if angle == 90:
            test_img = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            test_img = cv2.rotate(gray, cv2.ROTATE_180)
        elif angle == 270:
            test_img = cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            test_img = gray
        _, thresh = cv2.threshold(test_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        h_proj = np.sum(thresh, axis=1)
        peaks = np.where(h_proj > np.mean(h_proj))[0]
        if len(peaks) < 5:
            confidences.append(0)
            continue
        lines, start = [], peaks[0]
        for i in range(1, len(peaks)):
            if peaks[i] != peaks[i-1] + 1:
                lines.append((start, peaks[i-1]))
                start = peaks[i]
        lines.append((start, peaks[-1]))
        line_asymmetries = []
        for start_y, end_y in lines:
            if end_y - start_y < 5: continue
            line_img = thresh[start_y:end_y, :]
            h, _ = line_img.shape
            midpoint = h // 2
            top_half_sum = np.sum(line_img[0:midpoint, :])
            bottom_half_sum = np.sum(line_img[midpoint:, :])
            if bottom_half_sum > 0 and top_half_sum > 0:
                line_asymmetries.append(bottom_half_sum > top_half_sum)
        if line_asymmetries:
            confidences.append(np.mean(line_asymmetries))
        else:
            confidences.append(0)
    best_angle = angles[np.argmax(confidences)]
    if best_angle == 90: return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif best_angle == 180: return cv2.rotate(image, cv2.ROTATE_180)
    elif best_angle == 270: return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image

def verify_and_correct_orientation_tesseract(image: np.ndarray) -> np.ndarray:
    import cv2, pytesseract
    h, w = image.shape[:2]
    target_width = 1500
    if w > target_width:
        scale = target_width / w
        small_img = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small_img = image
    osd = pytesseract.image_to_osd(small_img, output_type=pytesseract.Output.DICT)
    rotation = osd.get('rotate', 0)
    if rotation != 0:
        if rotation == 90: return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180: return cv2.rotate(image, cv2.ROTATE_180)
        elif rotation == 270: return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image

def preprocess_image(image_pil: Image.Image, zoom_factor: float = 2.0) -> np.ndarray:
    """
    Accepts PIL.Image (RGB). Returns BGR numpy image (cv2-style) after orientation, deskew, crop, and upscaling.
    """
    import cv2
    image = np.array(image_pil)[:, :, ::-1].copy()  # PIL RGB -> BGR
    initial_orientation = correct_orientation_advanced_cv(image)
    img_oriented = verify_and_correct_orientation_tesseract(initial_orientation)
    gray_oriented = cv2.cvtColor(img_oriented, cv2.COLOR_BGR2GRAY)
    _, thresh_oriented = cv2.threshold(gray_oriented, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    from scipy.ndimage import interpolation as inter
    coarse_scores = []
    coarse_angles = np.arange(-5, 5.1, 1)
    for angle in coarse_angles:
        rotated = inter.rotate(thresh_oriented, angle, reshape=False, order=0)
        projection = np.sum(rotated, axis=1)
        coarse_scores.append((np.var(projection), angle))
    _, coarse_angle = max(coarse_scores, key=lambda x: x[0])
    fine_search_start = coarse_angle - 1
    fine_search_end = coarse_angle + 1
    fine_scores = []
    fine_angles = np.arange(fine_search_start, fine_search_end + 0.1, 0.1)
    for angle in fine_angles:
        rotated = inter.rotate(thresh_oriented, angle, reshape=False, order=0)
        projection = np.sum(rotated, axis=1)
        fine_scores.append((np.var(projection), angle))
    _, best_angle = max(fine_scores, key=lambda x: x[0])
    (h, w) = img_oriented.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    abs_cos, abs_sin = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(h * abs_sin + w * abs_cos)
    new_h = int(h * abs_cos + w * abs_sin)
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    deskewed = cv2.warpAffine(img_oriented, M, (new_w, new_h),
                                 flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(255, 255, 255))
    deskewed_gray = cv2.cvtColor(deskewed, cv2.COLOR_BGR2GRAY)
    _, deskewed_thresh = cv2.threshold(deskewed_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated = cv2.dilate(deskewed_thresh, kernel, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return deskewed
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    tight_crop = deskewed[max(0, y):y + h, max(0, x):x + w]
    final_result = cv2.resize(tight_crop, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_LINEAR)
    return final_result


class LineExtractor:
    def __init__(self, inverted_image: np.ndarray):
        import cv2, numpy as np
        self.inverted_image = inverted_image
        self.vertical_lines_eroded_image = None
        self.horizontal_lines_eroded_image = None
        self.combined_image = None

    def erode_vertical_lines(self):
        import cv2, numpy as np
        vertical_kernel = np.array([[1], [1], [1], [1], [1], [1]])
        self.vertical_lines_eroded_image = cv2.erode(self.inverted_image, vertical_kernel, iterations=10)
        self.vertical_lines_eroded_image = cv2.dilate(self.vertical_lines_eroded_image, vertical_kernel, iterations=13)

    def erode_horizontal_lines(self):
        import cv2, numpy as np
        hor = np.array([[1,1,1,1,1,1]])
        self.horizontal_lines_eroded_image = cv2.erode(self.inverted_image, hor, iterations=10)
        self.horizontal_lines_eroded_image = cv2.dilate(self.horizontal_lines_eroded_image, hor, iterations=13)

    def combine_eroded_images(self):
        import cv2
        self.combined_image = cv2.add(self.vertical_lines_eroded_image, self.horizontal_lines_eroded_image)

# projection-based repair function (keeps your implementation)
def repair_table_with_projection(image_with_white_lines: np.ndarray):
    import cv2, numpy as np
    from scipy.signal import find_peaks
    _, binary_image = cv2.threshold(image_with_white_lines, 127, 255, cv2.THRESH_BINARY)
    h, w = binary_image.shape
    horizontal_projection = np.sum(binary_image, axis=0) // 255
    peaks_x, _ = find_peaks(horizontal_projection, prominence=h*0.2)
    vertical_projection = np.sum(binary_image, axis=1) // 255
    peaks_y, _ = find_peaks(vertical_projection, prominence=w*0.2)
    reconstructed_grid = np.zeros_like(binary_image)
    line_color = 255
    line_thickness = 2
    for x in peaks_x:
        cv2.line(reconstructed_grid, (x, 0), (x, h), line_color, line_thickness)
    for y in peaks_y:
        cv2.line(reconstructed_grid, (0, y), (w, y), line_color, line_thickness)
    return reconstructed_grid, horizontal_projection, vertical_projection, peaks_x, peaks_y

class ContourTableDetector:
    def __init__(self, image_with_text, original_image, table_grid_image=None):
        import numpy as np
        import cv2
        from PIL import Image

        self.thresholded_image = image_with_text
        self.original_image = original_image
        self.table_grid_image = table_grid_image
        self.dilated_image = None
        self.contours = None
        self.bounding_boxes = []
        self.rows = []
        self.table_grid = []
        self.mean_height = 0
        self.table = []
        self.confidence_table = []

        # OCR engine init
        self.ocr_engine = None
        try:
            from paddleocr import PaddleOCR
            self.ocr_engine = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
            print(" PaddleOCR engine loaded.")
        except Exception:
            self.ocr_engine = None
            print(" Warning: PaddleOCR is not installed or paddlepaddle missing.")

        try:
            import pytesseract
            self.tesseract_available = True
        except Exception:
            self.tesseract_available = False

        # Super-res init (OpenCV DNN SR)
        self.super_res_model = None
        self.SUPER_RES_AVAILABLE = False
        try:
            sr = cv2.dnn_superres.DnnSuperResImpl_create()
            model_path = os.path.join(os.getcwd(), "EDSR_x4.pb")
            sr.readModel(model_path)
            sr.setModel("edsr", 4)
            self.super_res_model = sr
            self.SUPER_RES_AVAILABLE = True
            print(" AI Super-Resolution model loaded successfully.")
        except Exception as e:
            self.SUPER_RES_AVAILABLE = False
            print(" Warning: Could not load Super-Resolution model (cv2.dnn_superres unavailable).", e)

    def _build_grid_from_lines(self):
        import cv2
        if self.table_grid_image is None:
            print("INFO: No table grid image provided. Skipping line-based detection.")
            return

        print("INFO: Attempting Primary Strategy: Detecting cells from grid lines...")
        contours, hierarchy = cv2.findContours(self.table_grid_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cell_boxes = []
        if hierarchy is None:
            print("WARNING: No contour hierarchy found in grid image.")
            return

        for i, cnt in enumerate(contours):
            child_index = hierarchy[0][i][2]
            if child_index != -1:
                current_child_index = child_index
                while current_child_index != -1:
                    x, y, w, h = cv2.boundingRect(contours[current_child_index])
                    if w > 10 and h > 10:
                        cell_boxes.append((x, y, w, h))
                    current_child_index = hierarchy[0][current_child_index][0]

        if not cell_boxes:
            print("WARNING: Primary strategy could not detect any cells from grid lines.")
            return

        sorted_boxes = sorted(cell_boxes, key=lambda box: box[1])
        rows = []
        if not sorted_boxes:
            return
        current_row = [sorted_boxes[0]]
        for box in sorted_boxes[1:]:
            if abs(box[1] - current_row[-1][1]) < 20:
                current_row.append(box)
            else:
                rows.append(current_row)
                current_row = [box]
        rows.append(current_row)
        for row in rows:
            row.sort(key=lambda box: box[0])
        self.table_grid = rows
        print(f"Primary Strategy Successful: Found a grid with {len(self.table_grid)} rows.")

    def _build_grid_from_text_blobs(self):
        import cv2
        import numpy as np
        print("INFO: Attempting Fallback Strategy: Detecting cells from text blobs...")
        kernel = np.ones((1, 20), np.uint8)
        self.dilated_image = cv2.dilate(self.thresholded_image, kernel, iterations=3)
        self.contours, _ = cv2.findContours(self.dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.bounding_boxes = []
        for contour in self.contours:
            if cv2.contourArea(contour) > 50:
                self.bounding_boxes.append(cv2.boundingRect(contour))
        heights = [h for (_, _, _, h) in self.bounding_boxes]
        self.mean_height = np.mean(heights) if heights else 0
        self.bounding_boxes = sorted(self.bounding_boxes, key=lambda box: box[1])
        if not self.bounding_boxes:
            return
        row_tolerance = self.mean_height / 1.5
        self.rows = []
        current_row = [self.bounding_boxes[0]]
        for box in self.bounding_boxes[1:]:
            previous_box_y_center = current_row[-1][1] + current_row[-1][3] / 2
            current_box_y_center = box[1] + box[3] / 2
            if abs(current_box_y_center - previous_box_y_center) < row_tolerance:
                current_row.append(box)
            else:
                self.rows.append(current_row)
                current_row = [box]
        self.rows.append(current_row)
        for i in range(len(self.rows)):
            self.rows[i] = sorted(self.rows[i], key=lambda box: box[0])
        self._split_merged_cells()
        print(f" Fallback Strategy Successful: Found a grid with {len(self.table_grid)} rows.")

    def _split_merged_cells(self):
        if not self.rows:
            return
        reference_row_index = max(range(len(self.rows)), key=lambda i: len(self.rows[i]))
        reference_row = self.rows[reference_row_index]
        num_columns = len(reference_row)
        column_starts = [box[0] for box in reference_row]
        self.table_grid = []
        for row in self.rows:
            grid_row = [None] * num_columns
            for box in row:
                x, y, w, h = box
                start_col = -1
                for i in range(num_columns - 1, -1, -1):
                    if x >= column_starts[i] - (w * 0.2):
                        start_col = i
                        break
                if start_col == -1:
                    continue
                grid_row[start_col] = box
            self.table_grid.append(grid_row)

    def crop_each_bounding_box_and_ocr(self):
        import numpy as np
        import cv2
        if not self.table_grid:
            print("WARNING: Primary strategy failed. Executing fallback strategy.")
            self._build_grid_from_text_blobs()
        if not self.table_grid:
            print("ERROR: Both detection strategies failed. Could not form a table.")
            return
        print("\nINFO: Starting OCR process on the final grid...")
        for row_idx, grid_row in enumerate(self.table_grid):
            current_text_row = []
            current_confidence_row = []
            for cell_content in grid_row:
                if cell_content is None:
                    current_text_row.append("")
                    current_confidence_row.append("")
                    continue
                x, y, w, h = cell_content
                padding = 5
                cropped_image = self.original_image[max(0, y - padding):y + h + padding, max(0, x - padding):x + w + padding]
                if cropped_image.size > 0:
                    text_result, conf_result = self._get_result_from_paddleocr(cropped_image)
                    if not text_result.strip():
                        text_result, conf_result = self._strong_ocr_pass(cropped_image)
                    if not text_result.strip() and self.SUPER_RES_AVAILABLE:
                        text_result, conf_result = self._super_resolution_ocr_pass(cropped_image)
                    current_text_row.append(text_result.strip())
                    current_confidence_row.append(conf_result if text_result.strip() else "")
                else:
                    current_text_row.append("")
                    current_confidence_row.append("")
            self.table.append(current_text_row)
            self.confidence_table.append(current_confidence_row)
            print(f"   > Processed row {row_idx + 1}/{len(self.table_grid)}")
        print(" OCR process complete.")

    def _get_result_from_paddleocr(self, image):
        try:
            if not self.ocr_engine:
                return "", 0.0
            res = self.ocr_engine.ocr(image, cls=True)
            if not res or not res[0]:
                return "", 0.0
            texts = [line[1][0] for line in res[0]]
            scores = [line[1][1] for line in res[0]]
            full_text = " ".join(texts)
            avg_score = float(np.mean(scores)) if scores else 0.0
            return full_text, avg_score
        except Exception as e:
            print("   ERROR: PaddleOCR pass failed:", e)
            return "", 0.0

    def _strong_ocr_pass(self, image):
        import cv2
        import numpy as np
        print("   INFO: Tier 1 failed, attempting Tier 2 (Stronger Preprocessing)...")
        if image is None or not self.ocr_engine:
            return "", 0.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        scale = max(1.0, 150 / h)
        upscaled = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)
        denoised = cv2.fastNlMeansDenoising(upscaled, h=30)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrasted = clahe.apply(denoised)
        _, binarized = cv2.threshold(contrasted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return self._get_result_from_paddleocr(binarized)

    def _get_result_from_tesseract(self, image):
        import pytesseract
        from PIL import Image
        print("   INFO: Tier 2 failed, attempting Tier 3 (Tesseract)...")
        try:
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            text = pytesseract.image_to_string(pil_img, config=r'--oem 3 --psm 6')
            return text, 0.0
        except Exception as e:
            print("   ERROR: Tesseract pass failed:", e)
            return "", 0.0

    def _super_resolution_ocr_pass(self, image):
        print("   INFO: Tier 3 failed, attempting Tier 4 (AI Super-Resolution)...")
        try:
            sr_image = self.super_res_model.upsample(image)
            return self._strong_ocr_pass(sr_image)
        except Exception as e:
            print(f"   ERROR: Super-resolution pass failed: {e}")
            return "", 0.0

    def generate_csv_file(self, filename="output.csv"):
        import csv
        with open(filename, "w", encoding="utf-8", newline='') as f:
            writer = csv.writer(f)
            writer.writerows(self.table)
        print(f"\n Successfully generated text file: {filename}")

    def generate_confidence_csv_file(self, filename="confidence_output.csv"):
        import csv
        with open(filename, "w", encoding="utf-8", newline='') as f:
            writer = csv.writer(f)
            writer.writerows(self.confidence_table)
        print(f" Successfully generated confidence score file: {filename}")

    def get_image_with_final_grid(self):
        import cv2
        image_with_boxes = self.original_image.copy()
        for row_idx, grid_row in enumerate(self.table_grid):
            for col_idx, cell_content in enumerate(grid_row):
                if isinstance(cell_content, tuple) and len(cell_content) == 4:
                    x, y, w, h = cell_content
                    cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(image_with_boxes, f"{row_idx},{col_idx}", (x + 5, y + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        return image_with_boxes

