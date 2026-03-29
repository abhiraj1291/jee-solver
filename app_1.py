import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
CORS(app)
from groq import Groq

app = Flask(__name__)
CORS(app)  # Allow Netlify to talk to Railway

# ── API Key from environment variable
API_KEY = os.environ.get("GROQ_API_KEY", "")

MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are an expert JEE (Joint Entrance Examination) tutor for Physics, Chemistry, and Mathematics.

When solving any question:
1. Identify the topic/concept involved
2. Write the relevant formula(s) or theorem(s)
3. Show a clear, numbered step-by-step solution
4. State the FINAL ANSWER clearly on its own line
5. Give a short tip or mention a common mistake to avoid
6. Also give specific links appropriate to the question like YouTube, Scribd or Wikipedia.
7. Also provide useful links from the internet. Also check that content is available or not specially YouTube because the link you give is not available on platform.
Be concise, accurate, and use proper scientific notation.
For MCQ questions, identify the correct option AND briefly explain why the others are wrong."""

client = Groq(api_key=API_KEY)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "").strip()
    history = data.get("history", [])

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history[:-1]:
        if turn["role"] in ("user", "assistant"):
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_message})

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=2048,
        )
        reply = response.choices[0].message.content
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
