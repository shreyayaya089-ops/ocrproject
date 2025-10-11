import os
import csv
import time
from ocr_utils import process_image_and_get_plate

# ------------------------------------
# 1️⃣ Define test folder and CSV file
# ------------------------------------
TEST_FOLDER = 'test_images'
GT_CSV = os.path.join(TEST_FOLDER, 'ground_truth.csv')

# ------------------------------------
# 2️⃣ Load the ground truth CSV
# ------------------------------------
rows = []
with open(GT_CSV, newline='') as f:
    reader = csv.DictReader(f)
    for r in reader:
        # Clean up any extra spaces
        clean_row = {k.strip(): v.strip() for k, v in r.items()}
        rows.append(clean_row)

# ------------------------------------
# 3️⃣ Define Levenshtein distance
# ------------------------------------
def levenshtein(a, b):
    if a == b:
        return 0
    la, lb = len(a), len(b)
    dp = [[0]*(lb+1) for _ in range(la+1)]
    for i in range(la+1): dp[i][0] = i
    for j in range(lb+1): dp[0][j] = j
    for i in range(1, la+1):
        for j in range(1, lb+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return dp[la][lb]

# ------------------------------------
# 4️⃣ Run OCR and evaluate
# ------------------------------------
exact_matches = 0
char_acc_sum = 0.0
total = len(rows)
start_time = time.time()

for r in rows:
    filename = r['test_images']
    expected_plate = r['expected_plate'].strip().upper()

    image_path = os.path.join(TEST_FOLDER, filename)
    result = process_image_and_get_plate(image_path, debug=True, debug_dir="debug_out")
    pred_plate = (result.get('plate') or '').strip().upper()
    raw_text = result.get('raw', '')
    valid = result.get('valid', False)

    # Compute character-level accuracy
    lev_dist = levenshtein(pred_plate, expected_plate)
    max_len = max(len(pred_plate), len(expected_plate))
    char_level_acc = 1 - lev_dist / max_len if max_len > 0 else 0
    char_acc_sum += char_level_acc

    # Exact match
    exact_match = (pred_plate == expected_plate)
    if exact_match:
        exact_matches += 1

    print(f"{filename} → Pred: {pred_plate} | Expected: {expected_plate} | Match: {exact_match} | Char Acc: {char_level_acc*100:.2f}% | Raw: {raw_text}")

end_time = time.time()

# ------------------------------------
# 5️⃣ Print overall metrics
# ------------------------------------
print("\n========== EVALUATION SUMMARY ==========")
print(f"Total Images: {total}")
print(f"Exact Matches: {exact_matches} ({exact_matches / total * 100:.2f}%)")
print(f"Average Character Accuracy: {(char_acc_sum / total) * 100:.2f}%")
print(f"Average Time per Image: {(end_time - start_time) / total:.2f} sec")
print("=========================================")