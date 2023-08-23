import fitz
from PIL import Image
import pytesseract

pdf_document = fitz.open('testfiles/demo-report.pdf')

for page_number in range(pdf_document.page_count):
    page = pdf_document[page_number]
    blocks = page.get_text("blocks")  # Extracts text blocks
    
    # Convert page to pixmap (image) for processing with Tesseract
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    for block in blocks:
        # Extract coordinates
        x0, y0, x1, y1 = block[:4]
        
        # Define region of interest (ROI) using the coordinates
        roi = img.crop((x0, y0, x1, y1))
        
        # Process the ROI with Tesseract
        hocr_output = pytesseract.image_to_pdf_or_hocr(roi, extension='hocr')
        
        with open(f'output_page_{page_number + 1}_block_{blocks.index(block) + 1}.hocr', 'w', encoding='utf-8') as f:
            f.write(hocr_output.decode('utf-8'))

pdf_document.close()
