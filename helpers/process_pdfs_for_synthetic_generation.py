from datasets import load_dataset
import requests
import pdfplumber
from io import BytesIO
import os
import time

os.makedirs("pdfs", exist_ok = True)
def download_pdf(example, timeout=30):
    """
    Download PDF from URL and extract text using pdfplumber.
    
    Args:
        url (str): URL to download PDF from
        timeout (int): Timeout for download request in seconds
        
    Returns:
        str: Extracted text from PDF, or None if failed
    """
    time.sleep(0.1)
    url = example['url']
    if os.path.exists(f"pdfs/{example['id']}.pdf"):
        return {"pdf_path": f"pdfs/{example['id']}.pdf"}
    try:
        # Download PDF
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        # Check if content is PDF
        content_type = response.headers.get('content-type', '').lower()
        if 'pdf' not in content_type and not url.endswith('.pdf'):
            print(f"Warning: URL may not be a PDF. Content-Type: {content_type}")
        
        # save pdf
        with open(f"pdfs/{example['id']}.pdf", "wb") as f:
            f.write(response.content)
        return {"pdf_path": f"pdfs/{example['id']}.pdf"}
    except:
        return {"pdf_path": None}

def extract_text(example):
    pdf_path = example["pdf_path"]
    # Extract text using pdfplumber
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text_content = []
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text_content.append(f"{page_text}")
    except:
        return {"extracted_text": None}
        
    return {"extracted_text": "\n\n".join(text_content) if text_content else None}

papers = load_dataset("IVUL-KAUST/mextract_papers", split="train")
papers = papers.add_column("id", range(len(papers)))
papers = papers.shuffle(seed=42)
papers = papers.select(range(5000))
papers = papers.map(download_pdf, num_proc=16)
papers = papers.filter(lambda x: x['pdf_path'] is not None)
papers = papers.map(extract_text, num_proc=26)
papers = papers.filter(lambda x: x['extracted_text'] is not None)
papers.to_parquet("mextract_papers_with_text.parquet")