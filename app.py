from flask import Flask, jsonify, render_template, request, redirect, url_for, session
from flask_pymongo import PyMongo
from flask_bcrypt import Bcrypt
import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
import os
from collections import deque
from trie import Trie

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# MongoDB configuration
app.config["MONGO_URI"] = "mongodb+srv://ramakrishna:Anji%40178909@cluster0.ifqbcou.mongodb.net/temp?retryWrites=true&w=majority"
mongo = PyMongo(app)
bcrypt = Bcrypt(app)

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Error handling for loading files
MODEL_PATH = "chatbot.h5"
WORDS_PATH = "words.pkl"
CLASSES_PATH = "classes.pkl"

if os.path.exists(MODEL_PATH) and os.path.exists(WORDS_PATH) and os.path.exists(CLASSES_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    words = pickle.load(open(WORDS_PATH, 'rb'))
    classes = pickle.load(open(CLASSES_PATH, 'rb'))
else:
    raise FileNotFoundError("One or more required files not found. Please ensure that 'chatbot.h5', 'words.pkl', and 'classes.pkl' exist.")

# Define lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents data
with open('intents.json') as file:
    intents = json.load(file)

# Initialize chat history
CHAT_HISTORY_LIMIT = 50
chat_history = deque(maxlen=CHAT_HISTORY_LIMIT)
search_trie = Trie()

def add_to_chat_history(message, response):
    chat_history.append({'message': message, 'response': response})
    search_trie.insert(message)

# Functions for message processing and response prediction
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence.lower())
    return [lemmatizer.lemmatize(word) for word in sentence_words]

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = np.zeros(len(words), dtype=np.float32)
    for word in sentence_words:
        if word in words:
            bag[words.index(word)] = 1
    return bag

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]

    error_threshold = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > error_threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]

def get_response(intents_list, intents_json):
    if intents_list:
        max_prob_intent = intents_list[0]
        tag = max_prob_intent['intent']
        probability = float(max_prob_intent['probability'])

        if probability > 0.5:
            list_of_intents = intents_json['intents']
            for intent in list_of_intents:
                if intent['tag'] == tag:
                    result = random.choice(intent['responses'])
                    return result
    
    return "I'm sorry, I don't understand that."

@app.route('/')
def home():
    if 'username' in session:
        return render_template('index.html')
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        users = mongo.db.users
        login_user = users.find_one({'username': request.form['username']})

        if login_user and bcrypt.check_password_hash(login_user['password'], request.form['password']):
            session['username'] = request.form['username']
            return redirect(url_for('home'))
        return 'Invalid username/password combination'

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        users = mongo.db.users
        existing_user = users.find_one({'username': request.form['username']})

        if existing_user is None:
            hashpass = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')
            users.insert_one({'username': request.form['username'], 'password': hashpass})
            session['username'] = request.form['username']
            return redirect(url_for('home'))

        return 'That username already exists!'

    return render_template('register.html')

@app.route('/logout', methods=['POST'])
def logout():
    session.pop('username', None)
    return jsonify({'status': 'success'})

@app.route('/chat', methods=['POST'])
def chat():
    if 'username' not in session:
        return redirect(url_for('login'))
    global chat_history
    message = request.json['message']
    intents_list = predict_class(message)
    res = get_response(intents_list, intents)
    add_to_chat_history(message, res)
    return jsonify({'response': res})

@app.route('/history', methods=['GET'])
def history():
    if 'username' not in session:
        return redirect(url_for('login'))
    return jsonify(list(chat_history))

@app.route('/clear_history', methods=['POST'])
def clear_history():
    if 'username' not in session:
        return redirect(url_for('login'))
    global chat_history
    message_to_remove = request.json['message']
    chat_history = deque([item for item in chat_history if item['message'] != message_to_remove], maxlen=CHAT_HISTORY_LIMIT)
    return jsonify({'status': 'success'})


@app.route('/search_history', methods=['GET'])
def search_history():
    if 'username' not in session:
        return redirect(url_for('login'))
    query = request.args.get('query').lower()
    matching_messages = search_trie.get_words_with_prefix(query)
    results = [item for item in chat_history if item['message'].lower() in matching_messages]
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
