from ocr_utils import process_image_and_get_plate
 
# Path to your test image
img_path = r"C:\Users\prath\Desktop\ocrproject\test_images\sample1.jpeg"

# Run OCR
result = process_image_and_get_plate(img_path)
 
print("Raw OCR output:", result['raw'])
print("Processed Plate:", result['plate'])
print("Valid format?", result['valid'])
print("Confidence:", result['confidence'])