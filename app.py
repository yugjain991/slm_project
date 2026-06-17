from flask import Flask, render_template, request, jsonify
from main_light import get_bot_response

app = Flask(__name__)

@app.route("/")
def home():
    """Render the main chat interface."""
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """Handle incoming chat messages from the frontend."""
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "No message provided"}), 400

    import time
    start_time = time.time()
    
    user_message = data["message"]
    answer, source = get_bot_response(user_message)
    
    end_time = time.time()
    generation_time = round(end_time - start_time, 3)

    return jsonify({
        "answer": answer,
        "source": source,
        "time": generation_time
    })

if __name__ == "__main__":
    app.run(debug=True)
