from flask import Flask, render_template, request
from AUG import ask

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/answer', methods=['POST'])
def answer():
    question = request.form['question']
    api_key = request.form['api_key']
    title= request.form['title']
    # document_path = 'path/to/your/document.txt'  # Replace with the path to your document
    
    # # Load the document content from the file
    # with open(document_path, 'r', encoding='utf-8') as file:
    #     document_content = file.read()
    
    # Call your question answering function with the provided question and document
    
    answer = ask(question, api_key, title)

    
    return render_template('answer.html', question=question, answer=answer)

if __name__ == '__main__':
    app.run(debug=True)
