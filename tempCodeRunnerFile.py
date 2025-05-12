import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Load and preprocess data
lemmatizer = WordNetLemmatizer()

with open('intents1.json') as file:
    intents = json.load(file)

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',', '_']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern.lower())
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Prepare training data
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append(bag + output_row)

random.shuffle(training)
training = np.array(training)

train_x = training[:, :len(words)]
train_y = training[:, len(words):]

# Define and compile the model
model = Sequential([
    Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(len(train_y[0]), activation='softmax')
])

# Hyperparameter tuning
optimizer = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train the model
history = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Save the model in new format
model.save('chatbot.keras')

# Load model and data if needed
# model = tf.keras.models.load_model('chatbot.keras')
# words = pickle.load(open('words.pkl', 'rb'))
# classes = pickle.load(open('classes.pkl', 'rb'))

# Helper functions
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            return random.choice(i['responses'])

# Reinforcement mechanism
def get_feedback():
    feedback = input("Was this response helpful? (yes/no): ")
    return 1 if feedback.lower() == 'yes' else -1

# Reinforcement learning function with debugging and recompiling
def reinforce_learning(input_data, target_data):
    print("Input Data:", input_data)  # Debug: Print input data
    print("Target Data:", target_data)  # Debug: Print target data before training

    # Train the model with new data
    model.fit(np.array([input_data]), np.array([target_data]), epochs=5, verbose=0)

    # Recompile the model to ensure changes take effect immediately
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    print("Model updated with reinforcement learning.")

    # Run a test prediction after reinforcement
    test_prediction = model.predict(np.array([input_data]))[0]
    print("Model prediction after reinforcement:", test_prediction)

    # Save the updated model
    model.save('chatbot.keras')  # Save the model so updates are preserved
    print("Model saved after reinforcement.")

with open('intents1.json') as file:
    intents_json = json.load(file)  # Now this holds the full JSON data

# Updated chatbot_response function
def chatbot_response(text):
    intents = predict_class(text, model)  # Get the list of predicted intents
    response = get_response(intents, intents_json)  # Pass intents JSON data explicitly
    print(f"Bot: {response}")
    
    # Get feedback
    reward = get_feedback()
    
    # Reinforcement: Retrain model on positive feedback responses
    if reward > 0:
        input_data = bag_of_words(text, words)
        target_data = np.zeros(len(classes))
        target_data[classes.index(intents[0]['intent'])] = 1
        reinforce_learning(input_data, target_data)
    
    return response

# Running chatbot
print("Chatbot is ready! Type 'quit' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break
    chatbot_response(user_input)