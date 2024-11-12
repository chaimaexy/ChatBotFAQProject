from flask import Flask, render_template, request, jsonify
import random
import string
import nltk
nltk.download('punkt')  # Téléchargement de punkt pour le tokeniseur
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import pickle

app = Flask(__name__)

# Load your pre-processed tokens
try:
    with open('sent_tokens.pkl', 'rb') as f:
        sent_tokens = pickle.load(f)
except FileNotFoundError:
    print("sent_tokens.pkl not found. Please create it first.")
    sent_tokens = []  # Si le fichier n'existe pas, initialise une liste vide

# Initialize the lemmatizer
lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

def response(user_response):
    robo_response = ''
    sent_tokens.append(user_response)
    # Correction de l'avertissement lié au token_pattern
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english', token_pattern=None)
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]

    if(req_tfidf == 0):
        robo_response = robo_response + "I am sorry! I don't understand you."
    else:
        robo_response = robo_response + sent_tokens[idx]
    
    sent_tokens.remove(user_response)
    return robo_response

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/get_response", methods=["POST"])
def get_response():
    user_input = request.form["msg"]
    user_input = user_input.lower()

    if user_input == 'bye':
        return jsonify({"response": "Bye! Take care.."})

    if greeting(user_input) is not None:
        return jsonify({"response": greeting(user_input)})

    else:
        return jsonify({"response": response(user_input)})

if __name__ == "__main__":
    app.run(debug=True)
