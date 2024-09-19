
import os
from PyPDF2 import PdfReader

class PDFProcessor:
    def __init__(self, pdf_folder):
        self.pdf_folder = pdf_folder

    def extract_text_from_pdfs(self):
        all_texts = []
        # catch if no pdf files in the folder
        if not os.listdir(self.pdf_folder):
            print("No PDF files found in the folder.")
            return
        for pdf_file in os.listdir(self.pdf_folder):
            if pdf_file.endswith('.pdf'):
                reader = PdfReader(f"{self.pdf_folder}/{pdf_file}")
                text = ''.join([page.extract_text() for page in reader.pages])
                all_texts.append(text)
        return all_texts
    