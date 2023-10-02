import PyPDF2
import os
from PyPDF2 import PdfReader
import re

file_path = "/Users/adrienkwong/Downloads/FastLegal files/FastLegal - LlamaIndex + Streamlit/data/DCPI002109_2013.pdf"

reader = PdfReader(file_path)
headings_and_text = {}

# Define a regular expression pattern for headings (including italicized headings)
heading_pattern = r'^\s*(?:\*{1,2}|_{1,2})\s*[^*_]+(?:\*{1,2}|_{1,2})\s*$'






unwanted_chunk = "A \n \n \n \nB \n \n \n \nC \n \n \n \nD \n \n \n \nE \n \n \n \nF \n \n \n \nG \n \n \n \nH \n \n \n \nI \n \n \n \nJ \n \n \n \nK \n \n \n \nL \n \n \n \nM \n \n \n \nN \n \n \n \nO \n \n \n \nP \n \n \n \nQ \n \n \n \nR \n \n \n \nS \n \n \n \nT \n \n \n \nU \n \n \n \nV A \n \n \n \nB \n \n \n \nC \n \n \n \nD \n \n \n \nE \n \n \n \nF \n \n \n \nG \n \n \n \nH \n \n \n \nI \n \n \n \nJ \n \n \n \nK \n \n \n \nL \n \n \n \nM \n \n \n \nN \n \n \n \nO \n \n \n \nP \n \n \n \nQ \n \n \n \nR \n \n \n \nS \n \n \n \nT \n \n \n \nU \n \n \n \nV"


for page in reader.pages:
    page_text = page.extract_text()
    cleaned_page_text = page_text.replace(unwanted_chunk, "")
    lines = cleaned_page_text.split('\n')

    current_heading = None
    current_text = []

    for line in lines:
        if re.match(heading_pattern, line):
            if current_heading:
                headings_and_text[current_heading] = '\n'.join(current_text)
            current_heading = line
            current_text = []
        else:
            current_text.append(line)

    if current_heading:
        headings_and_text[current_heading] = '\n'.join(current_text)

for heading, text in headings_and_text.items():
    print("Heading:", heading)
    # print("Text:", text)
    # print("---")