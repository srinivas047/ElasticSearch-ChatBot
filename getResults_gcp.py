from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import json
import yaml
import pandas as pd
import re
from rapidfuzz import fuzz
import pickle

import re, string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from flask_cors import CORS, cross_origin

client = Elasticsearch(
    "http://34.125.188.221:9200")

loaded_model = pickle.load(open("/home/chettisaub/ElasticSearchBot_gcp/FlaskApp/logisticModel.pkl", 'rb'))
tfidf_vectorizer = pickle.load(open("/home/chettisaub/ElasticSearchBot_gcp/FlaskApp/tfidf.pickle", 'rb'))

import nltk
nltk.download('omw-1.4')
 
#1. STOPWORD REMOVAL
def stopword(string):
    a= [i for i in string.split() if i not in stopwords.words('english')]
    return ' '.join(a)

#2. STEMMING
 
# Initialize the stemmer
snow = SnowballStemmer('english')
def stemming(string):
    a=[snow.stem(i) for i in word_tokenize(string) ]
    return " ".join(a)

#3. LEMMATIZATION
# Initialize the lemmatizer
wl = WordNetLemmatizer()
 
# This is a helper function to map NTLK position tags
# Full list is available here: https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocess(text):
    text = text.lower() #lowercase text
    text=text.strip()  #get rid of leading/trailing whitespace 
    text=re.compile('<.*?>').sub('', text) #Remove HTML tags/markups
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)  #Replace punctuation with space. Careful since punctuation can sometime be useful
    text = re.sub('\s+', ' ', text)  #Remove extra space and tabs
    text = re.sub(r'\[[0-9]*\]',' ',text) #[0-9] matches any digit (0 to 10000...)
    text=re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d',' ',text) #matches any digit from 0 to 100000..., \D matches non-digits
    text = re.sub(r'\s+',' ',text) #\s matches any whitespace, \s+ matches multiple whitespace, \S matches non-whitespace 
    
    return text

# Tokenize the sentence
def lemmatizer(string):
    word_pos_tags = nltk.pos_tag(word_tokenize(string)) # Get position tags
    a=[wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)] # Map the position tag and lemmatize the word/token
    return " ".join(a)

def finalpreprocess(string):
    return lemmatizer(stopword(preprocess(string)))

def classifier(query):

    cleaned = finalpreprocess(query)

    corpus = []
    corpus.append(cleaned)
    X_vector=tfidf_vectorizer.transform(corpus) #converting X_test to vector
    y_predict = loaded_model.predict(X_vector)

    if y_predict == 1:
        return "reddit"
    else:
        return "caring"

def getResults(query, CORE_NAME):

    if CORE_NAME not in ["chitchat", "caring", "friendly", "witty", "professional", "enthusiastic","dialogs"]:
        queryCol = 'body'
        returnCol = 'comment'
    else:
        queryCol = 'Question'
        returnCol = 'Answer'

    if query == "":
        return "Please type something"
    else:
        response = client.search(index=CORE_NAME, body=
            { 
                "query": {
                "match": {
                    queryCol:query
                }
            }
            }
            )

        score = float('-inf')
        if response:
            topResult = response['hits']['hits'][1]["_source"][returnCol]
        
            for document in response['hits']['hits']:
                try:
                    process_rec = re.sub(r"[^a-zA-Z0-9 ]", ' ', document['_source'][returnCol])
                    similarity = fuzz.ratio(query, process_rec)

                    if similarity > score:
                        score = max(similarity, similarity)
                        topResult = document['_source'][returnCol]
                except:
                    print("error handled")
                
            return topResult
        else:
            return "Sorry, I dont have answer to your question"


from flask import Flask, render_template, request
app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=['POST', 'GET'])
@cross_origin(supports_credentials=True)
def get_bot_response():
    query = request.args.get("msg")
    # "msg, topicflag, personalityflag, topic, personality"

    index = classifier(query)
    print("index ", index)
    return getResults(query, index)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9999)
