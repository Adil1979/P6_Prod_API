import re
import nltk
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, PorterStemmer
lemma = WordNetLemmatizer().lemmatize
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
#from sklearn.externals import joblib
import joblib
from joblib import load, dump

set_alpha = set(['a', 'b', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '-'])

tags_500 = open('list_tag_key_500.joblib', 'rb')
dict_tags_500 = joblib.load(tags_500)

def preprocessing(text):
  #HTML characters removing
  def remove_html(text):
    html_regex = re.compile('<.*?>') #Compile regular expresions
    return re.sub(html_regex, ' ', str(text)) # Replace regex by ' '
  text0 = remove_html(text)
  

  #URL removing
  def remove_url(text0):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.sub(url_regex, ' ', str(text0))
  text1 = remove_url(text0)
  

  #Punctuation removing
  def remove_punc(text1):
    clean_text = re.sub(r'[?|!|"|:|=|_|{|}|[|]|-|$|%|^|&|]',r' ',str(text1))
    clean_text = re.sub(r'[.|,|)|(|\|/|-|~|`|>|<|*|$|@|;|→]',r' ', clean_text)
    return clean_text
  text2 = remove_url(text1)
  

  def remove_other(text2):
    text = str(text2)
    clean_text = re.sub(r"[^a-zA-Z0-9#+-]", " ", text2.lower())
    return clean_text
  text3 = remove_other(text2)
  

  #Space removing
  def remove_space(text3):
    return ' '.join(str(text3).split())
  text4 = remove_space(text3)
  

  def remove_stopwords(text4):
    text4 = str(text4)
    text4 = " ".join(word for word in text4.split() if word not in gensim.parsing.preprocessing.STOPWORDS and word not in stop_words and word not in set_alpha )
    return text4
  text5 = remove_stopwords(text4)
  

  def tokenize(text5):
    tokens = [lemma(w) for w in text5.split()] #if w.isalpha()
    return tokens
  text6 = tokenize(text5)
  

  def to_string(text6):
    input_string = " ".join(text6)
    return input_string
  text7 = to_string(text6)
  
  def keep_tags(text7):
    text7 = str(text7)
    text7 = " ".join(tag for tag in text7.split() if tag in dict_tags_500)
    return text7
  text8 = keep_tags(text7)
  return text8

def tokenize(document):
  tokens = [lemma(w) for w in document.split()]         
  return tokens
