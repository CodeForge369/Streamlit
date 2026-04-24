import streamlit as st
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pickle


try:
    words = stopwords.words("english")
except:
    nltk.download("stopwords")
    words = stopwords.words("english")
stemmer = PorterStemmer()
st.set_page_config(page_title="News Classifier", layout="wide")
st.title('Smart News Classifier')
st.markdown('----')
# Open the file in binary mode for reading
with open('LogisticRegression.pickle', 'rb') as file:
    # Load the data from the file
    model = pickle.load(file)


#taking input from users
data  = st.text_area("Enter New for the Classification")

if st.button('Submit'):
    d = {"predict_news": [data]}
    df = pd.DataFrame(d)
    df['predict_news'] = list(map(lambda x: " ".join([i for i in x.lower().split() if i not in words]), df['predict_news']))
    df['predict_news'] = df['predict_news'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())
    
    predict_news_cat = model.predict(df['predict_news'])
    st.write(predict_news_cat)
    