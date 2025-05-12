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
from tensorflow.keras.optimizers import Adam

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

# Paths to files
MODEL_PATH = "chatbot.keras"
WORDS_PATH = "words.pkl"
CLASSES_PATH = "classes.pkl"

# Load model and data files
if os.path.exists(MODEL_PATH) and os.path.exists(WORDS_PATH) and os.path.exists(CLASSES_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    words = pickle.load(open(WORDS_PATH, 'rb'))
    classes = pickle.load(open(CLASSES_PATH, 'rb'))
else:
    raise FileNotFoundError("One or more required files not found. Please ensure that 'chatbot.keras', 'words.pkl', and 'classes.pkl' exist.")

# Define lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents data
with open('intents1.json') as file:
    intents = json.load(file)

# Initialize chat history and search trie
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

def get_response(intents_list):
    if intents_list:
        max_prob_intent = intents_list[0]
        tag = max_prob_intent['intent']
        for intent in intents['intents']:
            if intent['tag'] == tag:
                return random.choice(intent['responses'])
    return "I'm sorry, I don't understand that."

# Reinforcement learning function
def reinforce_learning(input_data, target_data):
    """
    This function performs reinforcement learning by training the model
    on the provided input and target data.
    """
    # Ensure the model is compiled before training
    if not model.optimizer:
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model on the new data
    model.fit(np.array([input_data]), np.array([target_data]), epochs=5, verbose=0)
    
    # Save the model after the reinforcement learning process
    model.save('chatbot_reinforced.keras')  # Save as a new version after reinforcement learning
    print("Reinforcement learning step completed successfully.")

# Flask routes
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
        username = request.form.get('username')
        password = request.form.get('password')
        email = request.form.get('email')

        if not username or not password or not email:
            return 'Username, password, and email are required!'

        # Basic email validation
        if '@' not in email or '.' not in email:
            return 'Please enter a valid email address!'

        # Check if username or email already exists
        existing_user = users.find_one({
            '$or': [
                {'username': username},
                {'email': email}
            ]
        })

        if existing_user:
            if existing_user.get('username') == username:
                return 'That username already exists!'
            return 'That email is already registered!'

        hashpass = bcrypt.generate_password_hash(password).decode('utf-8')
        
        # Insert user with email field
        try:
            users.insert_one({
                'username': username,
                'password': hashpass,
                'email': email
            })
            session['username'] = username
            return redirect(url_for('home'))
        except Exception as e:
            return f'An error occurred: {str(e)}'

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
    res = get_response(intents_list)
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

# Route for feedback and reinforcement learning
@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.json
    message = data['message']
    is_helpful = data['is_helpful']

    print(f"Feedback received: {'Positive' if is_helpful else 'Negative'} for message: '{message}'")

    if is_helpful:
        # Print confirmation of the reinforcement learning process starting
        print("Reinforcement learning initiated for this positive feedback.")

        input_data = bag_of_words(message)
        target_data = np.zeros(len(classes))
        intent = predict_class(message)[0]['intent']
        target_data[classes.index(intent)] = 1

        # Call the reinforcement learning function to update the model
        reinforce_learning(input_data, target_data)

        # Print confirmation after model training
        print("Reinforcement learning process completed and model updated.")
    else:
        print("No reinforcement learning as feedback was negative.")

    return jsonify({'status': 'Feedback received and processed.'})

if __name__ == '__main__':
    app.run(debug=True)
