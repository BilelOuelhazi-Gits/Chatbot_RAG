import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from flask import Flask, render_template, request, jsonify
from rag import ask_rag

app = Flask(__name__)

chat_history = []

@app.route("/")
def index():
    return render_template("chat_modern.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question")
    if not question:
        return jsonify({"error": "Question vide"}), 400

    answer, sources = ask_rag(question)
    chat_item = {"question": question, "answer": answer, "sources": sources}
    chat_history.append(chat_item)
    return jsonify(chat_item)

@app.route("/history", methods=["GET"])
def history():
    return jsonify(chat_history)

if __name__ == "__main__":
    app.run(debug=True)
