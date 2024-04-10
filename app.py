from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json
import gzip
import shutil

app = Flask(__name__)

# Decompress the model file
with gzip.open('language_translation_model.keras.gz', 'rb') as f_in:
    with open('language_translation_model.keras', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

# Load tokenizer
with open('tokenizer_eng.json', 'r') as f:
    tokenizer_eng_json = json.load(f)
    tokenizer_eng = tokenizer_from_json(tokenizer_eng_json)

with open('tokenizer_ger.json', 'r') as f:
    tokenizer_ger_json = json.load(f)
    tokenizer_ger = tokenizer_from_json(tokenizer_ger_json)

# Load model
loaded_model = load_model("language_translation_model.keras")
max_len = 53  # Define your maximum sequence length here

@app.route('/', methods=['GET', 'POST'])
def translate():
    if request.method == 'POST':
        user_input = request.form['english_sentence']

        # Tokenize and pad the input sequence
        input_seq = tokenizer_eng.texts_to_sequences([user_input])
        input_seq = pad_sequences(input_seq, maxlen=max_len, padding='post')

        # Predict the output sequence
        predicted_seq = loaded_model.predict(input_seq)

        predicted_text = []
        for word_index in np.argmax(predicted_seq, axis=-1)[0]:
            if word_index != 0:  # Ignore padding index
                word = tokenizer_ger.index_word.get(word_index, '<OOV>')
                predicted_text.append(word)

        # Join the predicted words into a sentence
        german_translation = ' '.join(predicted_text)
        return render_template('index.html', user_input=user_input, german_translation=german_translation)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
