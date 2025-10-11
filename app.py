from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file, abort
import os
import sqlite3
from ocr_utils import process_image_and_get_plate
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import io

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"

# ----------------------------
# Database connection helper
# ----------------------------
def get_db_connection():
    conn = sqlite3.connect("vehicle_data.db")
    conn.row_factory = sqlite3.Row
    return conn

# ----------------------------
# Home page
# ----------------------------
@app.route("/")
def index():
    return render_template("index.html")

# ----------------------------
# Image upload and OCR
# ----------------------------
@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file:
        return redirect(url_for("index"))

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    result = process_image_and_get_plate(filepath)
    plate = result["plate"]

    conn = get_db_connection()
    vehicle = conn.execute("SELECT * FROM vehicles WHERE plate = ?", (plate,)).fetchone()
    stolen = conn.execute("SELECT * FROM stolen_vehicles WHERE plate = ?", (plate,)).fetchone()
    challans = conn.execute("SELECT * FROM challans WHERE plate = ?", (plate,)).fetchall()

    # Insert scan into scans table
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO scans (image_path, raw_text, detected_plate) VALUES (?, ?, ?)",
        (filepath, result["raw"], plate)
    )
    scan_id = cur.lastrowid
    conn.commit()
    conn.close()

    return render_template(
        "result.html",
        raw=result["raw"],
        plate=plate,
        scan_id=scan_id,
        vehicle=vehicle,
        stolen=stolen,
        challans=challans
    )

# ----------------------------
# Manual correction route
# ----------------------------
@app.route('/save_correction', methods=['POST'])
def save_correction():
    data = request.get_json()
    scan_id = int(data.get('scan_id'))
    corrected = data.get('corrected_plate', '').upper().strip()

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "UPDATE scans SET corrected_plate=?, user_corrected=1 WHERE id=?",
        (corrected, scan_id)
    )
    conn.commit()
    conn.close()

    return jsonify(success=True, corrected=corrected)

# ----------------------------
# PDF report route
# ----------------------------
@app.route('/report/<int:scan_id>')
def report(scan_id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT image_path, uploaded_at, raw_text, detected_plate, corrected_plate
        FROM scans WHERE id=?
    """, (scan_id,))
    r = cur.fetchone()
    conn.close()

    if not r:
        abort(404)

    image_path, uploaded_at, raw_text, detected_plate, corrected_plate = r

    # --- Generate PDF ---
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 800, f"Vehicle OCR Report - Scan #{scan_id}")
    c.setFont("Helvetica", 11)
    c.drawString(50, 770, f"Uploaded: {uploaded_at}")
    c.drawString(50, 750, f"Raw OCR: {raw_text}")
    c.drawString(50, 730, f"Detected Plate: {detected_plate}")
    c.drawString(50, 710, f"Corrected Plate: {corrected_plate}")

    # Add image if available
    try:
        c.drawImage(image_path, 50, 450, width=400, preserveAspectRatio=True)
    except Exception as e:
        print("Image not added:", e)

    c.showPage()
    c.save()
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name=f'report_{scan_id}.pdf',
        mimetype='application/pdf'
    )
import qrcode
import io
from flask import send_file, abort

@app.route('/qr/<int:scan_id>')
def qr(scan_id):
    conn = get_db_connection()  # use your helper
    cur = conn.cursor()
    cur.execute("SELECT corrected_plate, detected_plate FROM scans WHERE id=?", (scan_id,))
    r = cur.fetchone()
    conn.close()

    if not r:
        abort(404)

    corrected, detected = r
    payload = f"http://127.0.0.1:5000/report/{scan_id}"

    # Generate QR image
    img = qrcode.make(payload)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)

    return send_file(buf, mimetype='image/png')
# ----------------------------
# Run Flask
# ----------------------------
if __name__ == "__main__":
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    app.run(debug=True)