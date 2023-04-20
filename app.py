import os
import re
import faiss
import json
import openai
import pdfplumber
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertForQuestionAnswering
os.environ["TOKENIZERS_PARALLELISM"] = "false"
app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)


# Set your OpenAI API key
openai.api_key = "your_openai_api_key"

model = SentenceTransformer("paraphrase-distilroberta-base-v1")
index = None
rules = []
rules_embeddings = []

# Index creation
index = faiss.IndexFlatL2(768)  # 768 is the dimension of the embeddings

@app.route("/")
def home():
    return send_from_directory(".", "index.html")

@app.route("/upload", methods=["POST"])
def upload():
    global rules, rules_embeddings, index

    if "file" not in request.files:
        return "No file part", 400

    file = request.files["file"]
    filename = secure_filename(file.filename)
    file.save(filename)

    rules = []
    with open(filename, "r") as f:
        rules = [line.strip() for line in f.readlines()]

    print(f"Rules uploaded: {len(rules)}")

    rules_embeddings = model.encode(rules)
    index = faiss.IndexFlatL2(768)
    index.add(np.array(rules_embeddings).astype("float32"))

    os.remove(filename)

    return jsonify({"message": "Rules uploaded and indexed successfully"})


@app.route("/query", methods=["POST"])
def query():
    global rules, rules_embeddings, index

    data = request.get_json()

    if "query" not in data:
        return "No query provided", 400

    if not rules:
        return "No rules available, please upload rules first", 400

    query = data["query"]
    top_k = data.get("top_k", 3)

    query_embedding = model.encode([query])
    actual_top_k = min(top_k, len(rules))
    D, I = index.search(np.array(query_embedding).astype("float32"), actual_top_k)

    print(f"Search results (Distances): {D}")
    print(f"Search results (Indices): {I}")

    relevant_rules = [rules[i] for i in I[0]]

    answer_openai = answer_query_openai_api(query, relevant_rules)

    return jsonify({
        "relevant_rules": relevant_rules,
        "answer_openai": answer_openai
    })
def answer_query_openai_api(query, relevant_rules):
    context = " ".join(relevant_rules)
    
    prompt = f"Question: {query}\nI need an accurate and compact answer from the following information:\n{context}\n\nAnswer:"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.5,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["\n"],
    )

    answer = response.choices[0].text.strip()
    print(f"answer={answer}")
    return answer


if __name__ == "__main__":
    app.run(debug=True)