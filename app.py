from flask import Flask, render_template, request, jsonify

from src.graph_builder import Graph_Builder
from src.generation import Generation

app = Flask(__name__)

# Load the vector database once when the server starts
chroma_db = Graph_Builder().graph_vector()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()

        question = data.get("query", "").strip()

        if not question:
            return jsonify({
                "response": "Please enter a question."
            })

        # Generate response
        response = Generation().generate_response(
            question=question,
            db=chroma_db
        )

        return jsonify({
            "response": response
        })

    except Exception as e:
        print(e)

        return jsonify({
            "response": f"Error: {str(e)}"
        }), 500


if __name__ == "__main__":
    app.run(debug=True)