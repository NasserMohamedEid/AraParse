from flask import Flask, render_template, request ,jsonify
from prompt_engine import output_function
from vector_store import generate_grammar_explanation
import re

app = Flask(__name__, template_folder='templates', static_folder='static')

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/parser", methods=["GET", "POST"])
def parser():
    sentences = []
    if request.method == "POST":
        sentence = request.form.get("sentence", "")
        if sentence:
            result = output_function(sentence)  # Process the sentence with your function
            # Split the result into sentences by punctuation marks
            sentences = re.split(r'[.!؟]\s*', result)
            # Remove any empty strings from the list (in case result ends with punctuation)
            sentences = [s for s in sentences if s]
        else:
            sentences = ["الرجاء إدخال جملة صحيحة!"]
    return render_template('parse_sentence.html', sentences=sentences)

@app.route("/chat", methods=["GET", "POST"])
def chat():
    if request.method == "GET":
        # Render the HTML template on initial page load
        return render_template('correct_sentence.html')
    elif request.method == "POST":
        data = request.get_json()
        sentence = data.get("sentence", "")
        
        # Process the sentence if it exists
        if sentence:
            result = generate_grammar_explanation(sentence)
            sentences = re.split(r'[.!؟]\s*', result)
            sentences = [s for s in sentences if s]  # Filter out empty entries
        else:
            sentences = ["الرجاء إدخال جملة صحيحة!"]
        
        # Send the response in JSON format for AJAX to handle dynamically
        return jsonify({"sentences": sentences})


if __name__ == "__main__":
    app.run(debug=True)
