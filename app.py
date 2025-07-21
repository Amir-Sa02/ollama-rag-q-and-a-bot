# File: app.py
# This version now manages a simple in-memory chat history.

from flask import Flask, render_template, request, jsonify
from rag_core import answer_with_rag

app = Flask(__name__)

# --- Chat History Management ---
# For this simple example, we'll use a global list to store the history.
# NOTE: In a real multi-user application, this should be handled with user sessions.
chat_history = []

@app.route("/")
def index():
    """Serves the main HTML page and resets the history for a new session."""
    global chat_history
    chat_history = [] # Reset history every time the page is reloaded
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """Handles the chat message and uses the history."""
    global chat_history
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"error": "Message is required"}), 400
    
    # --- Pass the current history to the RAG function ---
    bot_response = answer_with_rag(user_message, chat_history)
    
    # --- Update the history with the new exchange ---
    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": bot_response})
    
    return jsonify({"response": bot_response})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
