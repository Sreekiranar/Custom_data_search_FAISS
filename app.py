import os
import faiss
import json
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import BertTokenizer, BertForQuestionAnswering

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

model = SentenceTransformer("paraphrase-distilroberta-base-v1")
index = None
rules = []
rules_embeddings = []

# Initialize the summarization model and tokenizer
summarization_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
summarization_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

# Initialize the question-answering model and tokenizer
qa_model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
qa_tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

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
    summary = summarize_rules(relevant_rules)

    # Answer the query using the relevant rules
    answer = answer_query(query, relevant_rules)

    return jsonify({"relevant_rules": relevant_rules, "summary": summary, "answer": answer})


def summarize_rules(relevant_rules):
    inputs = summarization_tokenizer(relevant_rules, return_tensors="pt", truncation=True, padding=True, max_length=1024)
    summary_ids = summarization_model.generate(inputs["input_ids"], num_beams=4, max_length=150, early_stopping=True)
    summary = [summarization_tokenizer.decode(summary_id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for summary_id in summary_ids]
    return summary

def answer_query(question, relevant_rules):
    context = " ".join(relevant_rules)
    
    # Truncate the context if the combined length is too long
    max_context_length = 512 - len(qa_tokenizer.tokenize(question)) - 3
    context_tokens = qa_tokenizer.tokenize(context)
    if len(context_tokens) > max_context_length:
        context_tokens = context_tokens[:max_context_length]
        context = qa_tokenizer.convert_tokens_to_string(context_tokens)

    inputs = qa_tokenizer(question, context, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    start_positions = qa_model(**inputs).start_logits.argmax(dim=-1).item()
    end_positions = qa_model(**inputs).end_logits.argmax(dim=-1).item()
    answer = qa_tokenizer.decode(inputs["input_ids"][0][start_positions:end_positions+1], skip_special_tokens=True)
    return answer

if __name__ == "__main__":
    app.run(debug=True)