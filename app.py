# This is the backend web server, built with Flask.
from flask import Flask, render_template, request, jsonify

# --- Import the core RAG function from our other file ---
from rag_core import answer_with_rag

# Initialize the Flask application
app = Flask(__name__)

# --- API Endpoints ---

@app.route("/")
def index():
    """Serves the main HTML page."""
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """Handles the chat message from the user and returns the model's response."""
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"error": "Message is required"}), 400
    
    # --- Call the AI logic from rag_core.py ---
    bot_response = answer_with_rag(user_message)
    
    return jsonify({"response": bot_response})

# --- Run the Application ---
if __name__ == '__main__':
    # Runs the Flask app on http://127.0.0.1:5000
    app.run(debug=True, port=5000)