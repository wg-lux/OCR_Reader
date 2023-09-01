import unittest
from flask_testing import TestCase
from unittest.mock import patch
from io import BytesIO
import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
testingfiles = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(testingfiles)
from app import app
from PIL import Image
import io

def generate_mock_image(file_format='JPEG'):
    image = Image.new('RGB', (100, 100), color='red')
    image_io = io.BytesIO()
    image.save(image_io, format=file_format)
    image_io.seek(0)
    return image_io

class OCRReaderTests(TestCase):
    def create_app(self):
        app.config['TESTING'] = True # Ensure we're in testing mode
        return app

    def test_api_endpoint(self):
        filename = 'dateityptest.jpg'
        content_type = 'image/jpeg'
        content = generate_mock_image('JPEG')
        data = {'image': (content, filename, content_type)}
        response = self.client.post('/api_endpoint', data=data, content_type='multipart/form-data', follow_redirects=True)
        self.assertEqual(response.status_code, 200)

    def test_all_file_uploads(self):
        file_uploads = [
            ('dateityptest.jpg', 'image/jpeg', 'JPEG'),
            ('dateityptest.pdf', 'application/pdf', 'PDF'),
            ('demo-report.pdf', 'application/pdf', 'PDF'),
            ('dateityptest.png', 'image/png', 'PNG'),
            ('dateityptest.tiff', 'image/tiff', 'TIFF')
        ]

        for filename, content_type, file_format in file_uploads:
            with self.subTest(filename=filename, content_type=content_type, file_format=file_format):
                content = generate_mock_image(file_format)
                data = {'image': (content, filename)}
                response = self.client.post('/api_endpoint', data=data, content_type='multipart/form-data', follow_redirects=True)
                self.assertEqual(response.status_code, 200)


if __name__ == '__main__':
    unittest.main()