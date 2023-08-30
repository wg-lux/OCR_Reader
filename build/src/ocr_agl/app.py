from flask import Flask, request, jsonify
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import fitz
import os
import cv2
import numpy as np
from medcat.vocab import Vocab
from medcat.cdb import CDB
from medcat.cat import CAT
from medcat.meta_cat import MetaCAT
import spacy
import hunspell
import re
import spacy
from spacy.tokens import Doc
import json

def download_model(model_name):
    spacy.cli.download(model_name)


os.environ['TESSDATA_PREFIX'] = '/opt/homebrew/share/tessdata'
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
tessdata_dir_config = '--tessdata-dir "/opt/homebrew/share/tessdata"'

# Load vocab and cdb for medcat
try:
    vocab = Vocab()
    vocab.load(path="dictionaries/vocab.dat")
    cdb = CDB()
    cdb.load(path="dictionaries/cdb.dat")
    cat = CAT(cdb=cdb, vocab=vocab)
except:
    print("No Vocab and/or CDB for Medcat Usage provided. Please save them in src/ocr_agl/dictionaries/ to use medcat.")


# load a spaCy model
try:
    nlp = spacy.load("de_core_news_md")
except OSError:
    print("Model not found. Downloading...")
    download_model("de_core_news_md")
    nlp = spacy.load("de_core_news_md")

app = Flask(__name__)

@app.route('/run_script')
def run_script():
    return "Processing completed"

def replace_unwanted_characters(word):
    # Erlaube nur Zeichen a-z, A-Z, ä, ü, ö, Ä, Ü, Ö, ß, 1-9, . , ' / & % ! " < > + * # ( ) € und -
    allowed_chars = r"[^a-zA-ZäüöÄÜÖß0-9.,'/&%!\"<>+*#()\€_:-]"
    return re.sub(allowed_chars, '', word)

def correct_medical_terms_ocr_output(text):
    annotations = json.loads(cat.get_json(text))
    
    # Check the content of annotations
    #print(annotations) 
    
    if 'entities' in annotations:
        for ann in annotations['entities']:
            concept_name = ann['name']  # Get the name of the detected concept
            start = ann['start']  # Start position of the concept in the text
            end = ann['end']  # End position of the concept in the text
            text = text[:start] + concept_name + text[end:]
    
    return text

def spell_check(text, language='de_DE'):
    dic_file = 'dictionaries/de_DE_frami.dic'
    aff_file = 'dictionaries/de_DE_frami.aff'

    h = hunspell.HunSpell(dic_file, aff_file)
    corrected_words = []

    # Tokenize the text using spaCy
    doc = nlp(text)

    for token in doc:
        word = replace_unwanted_characters(token.text)
        if not h.spell(word):
            suggestions = h.suggest(word)
            corrected_word = suggestions[0] if suggestions else word
            corrected_words.append(corrected_word)
        else:
            corrected_words.append(word)

    corrected_text = ' '.join(corrected_words)
    return corrected_text

def process_image(image, use_mock=False):
    #print("Image size:", image.size)
    #print("Image format:", image.format)
    
    if image is None or image.size == 0:
        raise ValueError("Invalid or empty image passed to process_image")

    if use_mock:
        # Return a simple mock image     
        image = Image.new('RGB', (100, 100), color='red')
        return image
    
    # Convert image to OpenCV format
    image = np.array(image)
    
    
    # Convert image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Binarization
    #_,  image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Scale image
    desired_width = 5000
    aspect_ratio = image.shape[1] / image.shape[0] # width / height
    desired_height = int(desired_width / aspect_ratio)

    image = cv2.resize(image, (desired_width, desired_height), interpolation=cv2.INTER_AREA)    
    
    # Increase contrast using histogram equalization
    #image = cv2.equalizeHist(image)

    # Noise reduction with a Median filter
    #image = cv2.medianBlur(image, 1)

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
        img = Image.frombuffer("RGB", [pix.width, pix.height], pix.samples, "raw", "RGB", 0, 1)

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
    #print("Scaled coordinates:", scaled_coords)
    cropped_image = image.crop(scaled_coords)
    return cropped_image

def scale_coordinates(coords, image_size):
    # Convert the fractional coordinates into actual pixel values
    left = coords['left'] * image_size[0]
    top = coords['top'] * image_size[1]
    right = coords['right'] * image_size[0]
    bottom = coords['bottom'] * image_size[1]
    #print("scaled coordinates:", left, top, right, bottom)
    
    return (left, top, right, bottom)



def crop_image(image, coordinates):
    coordinates = scale_coordinates(coordinates, image.size)
    return image.crop(coordinates)


@app.route('/api_endpoint', methods=['POST'])
def api_endpoint():
    try:
        # Get the File(s)
        files = request.files.getlist('image')
        
        # Get the coordinates from request.form and parse them into a Python list
        coordinates_list = json.loads(request.form.get('coordinates', '[]'))
        

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
                idx = 0 
                # for debugging:
                #print("PDF data size:", len(pdf_data))
                #print("Number of images extracted:", len(images))

                for img in images:
                    if idx < len(coordinates_list):
                        coords = coordinates_list[idx]
                        
                        # for debugging:

                        #print("Coordinates:", coords)
                        #print("Image size before cropping:", img.size)

                        

                        img = crop_image(img, coords)
                        # for debugging:

                        #print("Image size after cropping:", img.size)


                    #print (images)
                    img = process_image(img)

                    if img is None or img.size == (0, 0):
                        raise ValueError("Failed to convert PDF page to image")
                    text = pytesseract.image_to_string(img, lang='deu')
                    extracted_text += text + ' '
                    idx += 1
            else:
                image = Image.open(file)
                
                # Check for coordinates and crop the image if they are provided
                if idx < len(coordinates_list):
                    coords = coordinates_list[idx]
                    image = crop_image(image, coords)

                image = process_image(image)
                extracted_text = pytesseract.image_to_string(image, lang='deu')
                extracted_text = correct_medical_terms_ocr_output(extracted_text)
                extracted_text = spell_check(extracted_text)

            result = {
                'filename': filename,
                'file_type': file_type,
                'extracted_text': extracted_text
            }
            results.append(result)

        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run()