import os
from flask import Flask, render_template, request, jsonify
from groq import Groq

app = Flask(__name__)

# ── Put your Groq API key in an environment variable (never hardcode in production)
API_KEY = os.environ.get("GROQ_API_KEY", "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")

MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are an expert JEE (Joint Entrance Examination) tutor for Physics, Chemistry, and Mathematics.

When solving any question:
1. Identify the topic/concept involved
2. Write the relevant formula(s) or theorem(s)
3. Show a clear, numbered step-by-step solution
4. State the FINAL ANSWER clearly on its own line
5. Give a short tip or mention a common mistake to avoid
6. Also give specific links appropriate to the question like YouTube, Scribd or Wikipedia.
7. Also provide useful links from the internet. dont give youtube link 
8. also tell about most repated question in jee related to that toipic.
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
    history = data.get("history", [])  # list of {role, content}

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    # Build full message list: system + history + new user message
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Add previous turns (exclude last user msg — it's sent separately)
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


@app.route("/clear", methods=["POST"])
def clear():
    # History is managed client-side, this is just a reset acknowledgment
    return jsonify({"status": "cleared"})


if __name__ == "__main__":
    app.run(debug=True)
