from flask import Flask, render_template, request, jsonify
from openai import OpenAI
import os
import math
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
client = OpenAI()

MODELS = [
    "gpt-3.5-turbo",     # GPT-3.5 model
    "gpt-3.5-turbo-16k", # GPT-3.5 with larger context
    "gpt-4",             # GPT-4 model (if you have access)
    "gpt-4o-mini",       # GPT-4 model with smaller context
]

def softmax(logits):
    exp_logits = [math.exp(logit) for logit in logits]
    sum_exp_logits = sum(exp_logits)
    return [exp_logit / sum_exp_logits for exp_logit in exp_logits]

def get_next_word_probabilities(input_text, model, temperature, top_k=20):
    try:
        if model.startswith("gpt-"):
            # Use chat completion for GPT models
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that predicts the next word in a sentence. Provide only the next word, no punctuation or explanation."},
                    {"role": "user", "content": f"Complete this sentence with the next single word, providing {top_k} different options:\n{input_text}"}
                ],
                max_tokens=top_k,
                n=1,
                temperature=temperature,
            )
            content = response.choices[0].message.content.strip()
            words = content.split()[:top_k]
            # For chat models, we don't have logprobs, so we'll use a simple ranking
            logits = [top_k - i for i in range(len(words))]
            probs = softmax(logits)
        else:
            # Use text completion for other models
            response = client.completions.create(
                model=model,
                prompt=f"Complete this sentence with the next single word, providing {top_k} different options:\n{input_text}\n\nNext words:",
                max_tokens=top_k,
                n=1,
                temperature=temperature,
                logprobs=top_k,
                stop=["\n"]
            )
            words = [token for token in response.choices[0].logprobs.tokens if token.strip()][:top_k]
            logprobs = [response.choices[0].logprobs.token_logprobs[i] for i, token in enumerate(response.choices[0].logprobs.tokens) if token.strip()][:top_k]
            probs = softmax(logprobs)
        
        words_and_probs = [
            {"word": word, "probability": prob} 
            for word, prob in zip(words, probs)
        ]
        
        return sorted(words_and_probs, key=lambda x: x['probability'], reverse=True)
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


@app.route('/')
def index():
    return render_template('index.html', models=MODELS)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_text = data['input_text']
    model = data['model']
    temperature = float(data['temperature'])
    
    predictions = get_next_word_probabilities(input_text, model, temperature)
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)