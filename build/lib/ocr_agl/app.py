import requests
from flask import Flask, request, jsonify
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import fitz
import os
import cv2
import numpy as np


os.environ['TESSDATA_PREFIX'] = '/opt/homebrew/share/tessdata'
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
tessdata_dir_config = '--tessdata-dir "/opt/homebrew/share/tessdata"'

app = Flask(__name__)
app.secret_key = "bananabread"

@app.route('/run_script')
def run_script():
    return "Processing completed"


def process_image(image):
    # Convert image to OpenCV format
    image = np.array(image)
    
    # Convert image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Scale image
    image = cv2.resize(image, (5000, 5000), interpolation = cv2.INTER_AREA)
    
    # Increase contrast using histogram equalization
    # image = cv2.equalizeHist(image)

    # Noise reduction with a Median filter
    # image = cv2.medianBlur(image, 1)

    # Skew correction
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # Convert back to PIL.Image format
    image = Image.fromarray(image)
      # Increase contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(0.9)
    
    sharpness = ImageEnhance.Sharpness(image)
    image = sharpness.enhance(0.3) 
    
            # Apply blurring filter
    image = image.filter(ImageFilter.GaussianBlur(radius=0.03))

    return image


def convert_pdf_to_images(pdf_data):
    doc = fitz.open(stream=pdf_data, filetype="pdf")
    images = []

    for page in doc:
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)

    return images

@app.route('/api_endpoint', methods=['POST'])
def api_endpoint():
    files = request.files.getlist('image')

    results = []

    for file in files:
        if file.filename == '':
            return jsonify({'error': 'No file uploaded.'}), 400

        filename = file.filename
        file_extension = filename.split('.')[-1].lower()

        mime_types = {
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'png': 'image/png',
            'gif': 'image/gif',
            'bmp': 'image/bmp',
            'tiff': 'image/tiff',
            'pdf': 'application/pdf',
        }

        # Get the corresponding MIME type based on the file extension
        file_type = mime_types.get(file_extension, 'application/octet-stream')
        file_type = file_type.split('/')[-1]

        if file_type not in ['jpg', 'jpeg', 'png', 'pdf', 'tiff']:
            return jsonify({'error': 'Invalid file type.'}), 400

        if file_type == 'pdf':
            pdf_data = file.read()
            images = convert_pdf_to_images(pdf_data)
            extracted_text = ''
            for image in images:
                image = process_image(image)
                text = pytesseract.image_to_string(image, lang='deu')
                extracted_text += text + ' '
        else:
            image = Image.open(file)
            image = process_image(image)
            extracted_text = pytesseract.image_to_string(image, lang='deu')

        result = {
            'filename': filename,
            'file_type': file_type,
            'extracted_text': extracted_text
        }
        results.append(result)

    return jsonify(results)

if __name__ == '__main__':
    app.run()