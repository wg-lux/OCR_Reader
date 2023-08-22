import fitz
from PIL import Image

import pytesseract

# Open the PDF file using fitz
pdf_document = fitz.open('testfiles/demo-report.pdf')

# Iterate through pages and convert each to an image
for page_number in range(pdf_document.page_count):
    page = pdf_document[page_number]
    
    # Convert page to pixmap (image)
    pix = page.get_pixmap()
    
    # Convert the pixmap to a PIL Image
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # Use pytesseract to get hOCR output
    hocr_output = pytesseract.image_to_pdf_or_hocr(img, extension='hocr')
    
    with open(f'output_page_{page_number + 1}.hocr', 'w', encoding='utf-8') as f:
        f.write(hocr_output.decode('utf-8'))

pdf_document.close()
