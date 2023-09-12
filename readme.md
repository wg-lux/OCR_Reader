
# OCR App with API

The OCR app is an application for extracting text from images and PDF files. It is built on Flask, a Python web framework, and utilizes the Tesseract OCR library and the PIL library for image processing.

# Features

API for the upload of images and PDF files for text extraction.
Support for various image formats such as JPG, JPEG, PNG and PDF.
Processing of PDF files by converting them into images and extracting text from the images.
API access to the same texts.

# Requirements

To run the app, the dependencies from requirements.txt must be installed:

Flask
pytesseract
Tesseract OCR
PIL (Python Imaging Library)
fitz
You can install the dependencies with pip by running the following command:

pip install -r requirements.txt

# Starting the Application

Run the app with the following command:

python app.py

The app will be started in test mode on http://localhost:5000.

# API Access Guide

For API usage, a request can be sent for example as Python code with the path of the image in the following form:

url = 'http://localhost:5000/api_endpoint'
image_path = '/image_path'
files = {'image': open(image_path, 'rb')}
response = requests.post(url, files=files)

Note: Make sure the app is running.

# Instructions

Make sure the app is running in your webbrowser. Since no content is put on the homepage you will see a server error. To use the API send a request like in the request.py file, supplying your path to the image .

# Note

Make sure that Tesseract OCR is installed on your system and the 'TESSDATA_PREFIX' environment variable is correctly set to the directory with the Tesseract language data.

# Rechtliches

Medizinische Daten werden mit MedCat klassifiziert.
Die Erstellung erfolgt unter Verwendung der maschinenlesbaren Fassung des Bundesinstituts für Arzneimittel und Medizinprodukte (BfArM).

Max Hild // AG Lux // 2023
