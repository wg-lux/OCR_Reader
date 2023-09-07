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

def crop_image(image, coords):
    """
    Crop the image based on normalized coordinates.

    Parameters:
    - image (PIL.Image.Image): The image to crop.
    - coords (dict): Dictionary with keys 'left', 'top', 'right', and 'bottom' as fractions of DIN A4 dimensions.

    Returns:
    - (PIL.Image.Image): Cropped image.
    """
    scaled_coords = scale_coordinates(coords, image.size)
    cropped_image = image.crop(scaled_coords)
    return cropped_image

def scale_coordinates(coords, image_size):
    """
    Scale coordinates based on the image size.

    Parameters:
    - coords (dict): Dictionary with keys 'left', 'top', 'right', and 'bottom' as fractions of DIN A4 dimensions.
    - image_size (tuple): (width, height) of the image in pixels.

    Returns:
    - (tuple): Scaled coordinates as (left, top, right, bottom) in pixels.
    """
    A4_WIDTH = 210  # in millimeters
    A4_HEIGHT = 297  # in millimeters

    width_scale = image_size[0] / A4_WIDTH
    height_scale = image_size[1] / A4_HEIGHT

    left = coords['left'] * width_scale
    top = coords['top'] * height_scale
    right = coords['right'] * width_scale
    bottom = coords['bottom'] * height_scale

    return (left, top, right, bottom)


def crop_image(image, coordinates):
    return image.crop((coordinates['left'], coordinates['top'], coordinates['right'], coordinates['bottom']))

@app.route('/api_endpoint', methods=['POST'])
def api_endpoint():
    files = request.files.getlist('image')
    coordinates_list = request.json.get('coordinates', [])

    results = []

    for idx, file in enumerate(files):
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

        file_type = mime_types.get(file_extension, 'application/octet-stream').split('/')[-1]
        if file_type not in ['jpg', 'jpeg', 'png', 'pdf', 'tiff']:
            return jsonify({'error': 'Invalid file type.'}), 400

        if file_type == 'pdf':
            pdf_data = file.read()
            images = convert_pdf_to_images(pdf_data)
            extracted_text = ''
            for img in images:
                if idx < len(coordinates_list):
                    coords = coordinates_list[idx]
                    img = crop_image(img, coords)
                img = process_image(img)
                text = pytesseract.image_to_string(img, lang='deu')
                extracted_text += text + ' '
        else:
            image = Image.open(file)
            
            # Check for coordinates and crop the image if they are provided
            if idx < len(coordinates_list):
                coords = coordinates_list[idx]
                image = crop_image(image, coords)

            image = process_image(image)
            extracted_text = pytesseract.image_to_string(image, lang='deu')

        result = {
            'filename': filename,
            'file_type': file_type,
            'extracted_text': extracted_text
        }
        results.append(result)

    return jsonify(results)
