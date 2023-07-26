import unittest
import requests
from flask_testing import TestCase
from flask import request
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
testingfiles = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(testingfiles)
from app import app


class OCRReaderTests(TestCase):
    def create_app(self):
        return app

    def test_api_endpoint(self):
        filename = 'tests/testfiles/dateityptest.jpg'
        content_type = 'image/jpeg'
        
        with open(os.path.join(project_root, filename), 'rb') as file:
            data = {'image': (file, filename)}
            response = self.client.post('/api_endpoint', data=data, content_type=content_type, follow_redirects=True)
            self.assertEqual(response.status_code, 200)
            # Add your assertions for the response data here

    def test_all_file_uploads(self):
        file_uploads = [
            ('tests/testfiles/dateityptest.jpg', 'image/jpeg'),
            ('tests/testfiles/dateityptest.pdf', 'application/pdf'),
            ('tests/testfiles/demo-report.pdf', 'application.pdf'),
            ('tests/testfiles/dateityptest.png', 'image/png'),
            ('tests/testfiles/dateityptest.tiff', 'image/tiff')
        ]

        for filename, content_type in file_uploads:
            with self.subTest(filename=filename, content_type=content_type):
                with open(os.path.join(testingfiles, filename), 'rb') as file:
                    data = {'image': (file, filename)}
                    response = self.client.post('/api_endpoint', data=data, content_type=content_type, follow_redirects=True)
                    self.assertEqual(response.status_code, 200)


if __name__ == '__main__':
    unittest.main()
