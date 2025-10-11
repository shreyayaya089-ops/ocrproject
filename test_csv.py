import csv
import os

TEST_FOLDER = 'test_images'
GT_CSV = os.path.join(TEST_FOLDER, 'ground_truth.csv')

rows = []
with open(GT_CSV, newline='') as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

print(rows)  # To see what it loaded