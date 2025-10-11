# ocr_utils.py
import cv2
import pytesseract
import numpy as np
import re
import os
 
# --- Optional: Path for Tesseract (if not in PATH) ---
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
 
# ----------------------------------------------------
# Confusion mappings (OCR character correction)
# ----------------------------------------------------
NUM_MAP = {'O': '0', 'Q': '0', 'D': '0', 'I': '1', 'L': '1', 'S': '5', 'B': '8', 'Z': '2', 'G': '6'}
LETTER_MAP = {'0': 'O', '1': 'I', '5': 'S', '8': 'B', '2': 'Z', '6': 'G', '4': 'A'}
GLOBAL_CONFUSIONS = {**NUM_MAP}
 
# ----------------------------------------------------
# Support for multiple Indian number plate formats
# ----------------------------------------------------
PLATE_PATTERNS = [
    r'^[A-Z]{2}\d{2}[A-Z]{1}\d{4}$',  # e.g., KA02A1234
    r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$',  # e.g., MH12AB1234
    r'^[A-Z]{2}\d{1,2}[A-Z]{1,2}\d{4}$',  # general fallback
]
PLATE_REGEXES = [re.compile(p) for p in PLATE_PATTERNS]
 
# ----------------------------------------------------
# Utility functions
# ----------------------------------------------------
def _clean_text(s):
    return re.sub(r'[^A-Z0-9]', '', s.upper() if s else '')
 
def _matches_plate_format(s):
    for reg in PLATE_REGEXES:
        if reg.match(s):
            return True
    return False
 
def position_aware_correction(s):
    s = _clean_text(s)
    s2 = ''.join([GLOBAL_CONFUSIONS.get(ch, ch) for ch in s])
    if _matches_plate_format(s2):
        return s2
    return s2
 
# ----------------------------------------------------
# OCR helper
# ----------------------------------------------------
def ocr_with_confidence(img_gray, psm=7):
    config = f"--psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    data = pytesseract.image_to_data(img_gray, config=config, output_type=pytesseract.Output.DICT)
    texts = data.get('text', [])
    confs = data.get('conf', [])
 
    words, conf_vals = [], []
    for t, c in zip(texts, confs):
        t = t.strip()
        if t:
            words.append(t)
            try:
                ci = float(c)
            except:
                ci = -1
            if ci >= 0:
                conf_vals.append(ci)
 
    full_raw = ' '.join(words)
    cleaned = _clean_text(full_raw)
    avg_conf = sum(conf_vals) / len(conf_vals) if conf_vals else 0.0
    return cleaned, avg_conf, full_raw
 
# ----------------------------------------------------
# Candidate detection
# ----------------------------------------------------
def detect_plate_candidates(img_bgr, min_area_ratio=0.001, debug_save_dir=None):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 15, 4)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
 
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    min_area = max(3000, w * h * min_area_ratio)
    candidates = []
 
    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        area = ww * hh
        if area < min_area:
            continue
        ar = ww / float(hh) if hh > 0 else 0
        if 1.4 < ar < 7.5:
            candidates.append((x, y, ww, hh, cnt, area))
 
    candidates = sorted(candidates, key=lambda x: x[5], reverse=True)
 
    if debug_save_dir:
        os.makedirs(debug_save_dir, exist_ok=True)
        for i, (x, y, ww, hh, _, _) in enumerate(candidates):
            crop = img_bgr[y:y+hh, x:x+ww]
            cv2.imwrite(os.path.join(debug_save_dir, f"candidate_{i}.jpg"), crop)
 
    return candidates
 
# ----------------------------------------------------
# Perspective warp
# ----------------------------------------------------
def warp_candidate(img_bgr, rect):
    rect_ma = cv2.minAreaRect(rect)
    box = cv2.boxPoints(rect_ma)
    box = np.int0(box)
 
    def order_pts(pts):
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        return np.array([
            pts[np.argmin(s)],
            pts[np.argmin(diff)],
            pts[np.argmax(s)],
            pts[np.argmax(diff)]
        ], dtype="float32")
 
    box_o = order_pts(box)
    (tl, tr, br, bl) = box_o
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxWidth = int(max(widthA, widthB))
    maxHeight = int(max(heightA, heightB))
 
    dst = np.array([[0, 0], [maxWidth-1, 0], [maxWidth-1, maxHeight-1], [0, maxHeight-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(box_o, dst)
    warped = cv2.warpPerspective(img_bgr, M, (maxWidth, maxHeight))
    return warped
 
# ----------------------------------------------------
# Main OCR function
# ----------------------------------------------------
def process_image_and_get_plate(image_path, debug=False, debug_dir="debug_out"):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return {"raw": "", "plate": "", "valid": False, "confidence": 0.0, "error": "file-not-found"}
 
    if debug:
        os.makedirs(debug_dir, exist_ok=True)
 
    candidates = detect_plate_candidates(img_bgr, debug_save_dir=(debug_dir if debug else None))
    trials = []
    psm_list = [6, 7, 8, 11, 12, 13]  # include multiline (psm 11, 12, 13)
    scales = [1.0, 1.5, 2.0]
 
    for idx, (x, y, w, h, cnt, area) in enumerate(candidates[:6]):
        try:
            warped = warp_candidate(img_bgr, cnt)
            crop_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        except Exception:
            crop_gray = cv2.cvtColor(img_bgr[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
 
        for scale in scales:
            if scale != 1.0:
                img_s = cv2.resize(crop_gray, None, fx=scale, fy=scale)
            else:
                img_s = crop_gray
 
            for psm in psm_list:
                cleaned, conf, raw = ocr_with_confidence(img_s, psm=psm)
                corrected = position_aware_correction(cleaned)
                score = len(cleaned) + conf / 10.0
                if _matches_plate_format(corrected):
                    score += 10
                trials.append({
                    "source": "candidate",
                    "idx": idx,
                    "psm": psm,
                    "scale": scale,
                    "raw": raw,
                    "cleaned": cleaned,
                    "corrected": corrected,
                    "conf": conf,
                    "score": score
                })
                if debug:
                    tag = f"cand{idx}_psm{psm}_s{scale}.png"
                    cv2.imwrite(os.path.join(debug_dir, tag), img_s)
 
    # Fallback: full image
    gray_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    for psm in psm_list:
        cleaned, conf, raw = ocr_with_confidence(gray_full, psm=psm)
        corrected = position_aware_correction(cleaned)
        score = len(cleaned) + conf / 10.0
        if _matches_plate_format(corrected):
            score += 5
        trials.append({
            "source": "full",
            "psm": psm,
            "raw": raw,
            "cleaned": cleaned,
            "corrected": corrected,
            "conf": conf,
            "score": score
        })
 
    if not trials:
        return {"raw": "", "plate": "", "valid": False, "confidence": 0.0, "error": "no-trials"}
 
    best = sorted(trials, key=lambda x: x['score'], reverse=True)[0]
    final_plate = best.get('corrected', '')
    final_raw = best.get('raw', '')
    conf = best.get('conf', 0.0)
    valid = _matches_plate_format(final_plate)
 
    result = {"raw": final_raw, "plate": final_plate, "valid": valid, "confidence": conf}
    if debug:
        result["debug_trials"] = trials[:5]
    return result
 