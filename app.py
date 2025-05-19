from flask import Flask, request, jsonify
import os
from pdf_processing import extract_text_from_pdf, extract_text_from_txt
from embeddings import generate_embeddings, create_faiss_index, search_faiss_index

app = Flask(__name__)

# Variáveis globais
index = None
texts = []

# Função para carregar e processar os arquivos PDF
def load_and_process_documents():
    global index, texts
    
    input_folder = "inputs/"
    files = os.listdir(input_folder)
    
    # Para cada arquivo na pasta "inputs", extrair texto
    for file_name in files:
        file_path = os.path.join(input_folder, file_name)
        
        if file_name.endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
        elif file_name.endswith('.txt'):
            text = extract_text_from_txt(file_path)
        else:
            continue
        
        texts.append(text)
    
    # Gera os embeddings dos textos
    embeddings = generate_embeddings(texts)
    
    # Cria o índice FAISS
    index = create_faiss_index(embeddings)

@app.route('/ask', methods=['POST'])
def ask():
    query = request.json.get('query')
    
    # Realiza a busca no índice FAISS
    distances, indices = search_faiss_index(index, query)
    
    # Recupera as respostas com base nos índices
    results = [texts[i] for i in indices[0]]
    
    return jsonify({
        'query': query,
        'results': results,
        'distances': distances[0].tolist()
    })

if __name__ == '__main__':
    # Carrega e processa os documentos quando iniciar o servidor
    load_and_process_documents()
    
    # Inicia o servidor Flask
    app.run(debug=True)
