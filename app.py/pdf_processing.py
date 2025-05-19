import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    """
    Função para extrair texto de um arquivo PDF.
    """
    document = fitz.open(pdf_path)
    text = ""
    
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text("text")
    
    return text

def extract_text_from_txt(txt_path):
    """
    Função para extrair texto de um arquivo de texto.
    """
    with open(txt_path, 'r', encoding='utf-8') as file:
        return file.read()
