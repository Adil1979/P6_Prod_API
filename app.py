import pandas as pd
import numpy as np
import os
import re
import nltk
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, PorterStemmer
lemma = WordNetLemmatizer().lemmatize
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.metrics import jaccard_similarity_score, make_scorer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
#from sklearn.externals import joblib
import joblib
from joblib import load, dump
from flask import Flask, render_template, request
from functions import preprocessing, tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
nltk.download('all')
nltk.download('wordnet')


path0=os.path.abspath(os.path.dirname(__file__))
file_0=os.path.join(path0, 'list_tag_key_500.joblib')
file_1=os.path.join(path0, 'reg_log_saved.joblib')


tags_500 = open(file_0, 'rb')
input_tags_500 = joblib.load(tags_500)

input_st_tags_500  = " ".join(input_tags_500)
input_df_tags_500 = pd.DataFrame([input_st_tags_500], columns=['question'])



# Load Model
API_model = open(file_1, 'rb')
pipeline = joblib.load(API_model)

app = Flask(__name__) # Creer app et charger les fonctionalités de Flask

# Home page
@app.route('/') 
def index():
    return render_template('index.html')
                                                    
@app.route('/tag_recommendation', methods=['POST'])
def tag_recommendation():
    
    # Appeler les Inputs de la page HTML dashboard
    question = request.form['question'] #request.args.get('question')
    tags_text = ''
    if question is not None:
        
        question = str(question)
        question_tag = preprocessing(question) 
        question_tag_df = pd.DataFrame([question_tag], columns=['question'])
        test_input_df = pd.concat([question_tag_df, input_df_tags_500], ignore_index=True)
        question_input = test_input_df['question']
        vectorizer = TfidfVectorizer(tokenizer = tokenize, stop_words = stop_words, max_features=355)
        X_tfidf = vectorizer.fit_transform(question_input).toarray()
        feature_names  = vectorizer.get_feature_names()
        X_test_question = pd.DataFrame(X_tfidf)
        X_test_question = X_test_question.iloc[0:1,:]
        X_test_question.columns = feature_names
          
        tags_num = pipeline.predict(X_test_question)
        mlb = MultiLabelBinarizer(classes=sorted(input_tags_500))
        mlb.fit(input_tags_500)
        tags_text = pd.concat([pd.Series(mlb.inverse_transform(tags_num), name='tags_num')],axis=1)
        tags_text = str(tags_text.values.tolist()).strip('[()]') 
        tags_text
        
       
    
    return render_template('recommendation.html', tags = tags_text)
                  

if __name__== '__main__': #Executer directement
    app.run(debug=True, port=4000) #Lancer le serveur local (localhost/adresse ip 127.0.0.1)



