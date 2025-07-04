from tkinter import *
import customtkinter as c
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('movie_reviews')
nltk.download('punkt_tab')
from nltk.corpus import stopwords, movie_reviews
from nltk.classify.util import accuracy
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
import string

def delete():
    txt.delete(0.0, "end")


stop_words = set(stopwords.words('english'))
punctuations = set(string.punctuation)

def preprocess(words):
    return [w.lower() for w in words if w.lower() not in stop_words and w not in punctuations]

documents = [(list(movie_reviews.words(fileid)),category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

def document_features(words):
    words=preprocess(words)
    return {word : True for word in words}

feature_sets=[(document_features(doc),category) for (doc,category) in documents]

train_size = int(len(feature_sets) * 0.8)
train_set, test_set = feature_sets[:train_size], feature_sets[train_size:]

classifier = NaiveBayesClassifier.train(train_set)

def predict_sentiment(text):
    tokens = word_tokenize(text)
    features = document_features(tokens)
    return classifier.classify(features)



c.set_appearance_mode('dark')
c.set_default_color_theme('dark-blue')

app = c.CTk()
app.title("text summarization")
app.geometry("400x400")

label1 = c.CTkLabel(app, text="Sentiment Analysis", font=("Arial", 40, "bold"),fg_color="blue")
label1.pack(pady=10)
txt = c.CTkTextbox(app, width=1000, height=80)
txt.pack(pady=20)

frame = c.CTkFrame(app)
frame.pack(pady=10)

def Analyze():
    txtout.delete(0.0,"end")
    sample_text = txt.get("0.0","end")
    if predict_sentiment(sample_text)=="pos":
        result = "Positive review"
    elif predict_sentiment(sample_text)=="neg":
        result ="Negative review"
    else:
        result = "Neutral review"
    txtout.insert("0.0",result)

del_btn = c.CTkButton(frame, text="Clear", command=delete)
sum_btn = c.CTkButton(frame, text="Analyze", command=Analyze)
del_btn.grid(row=0, column=0)
sum_btn.grid(row=0, column=1)

frame1=c.CTkFrame(app)
frame1.pack(pady=10)

txtout = c.CTkTextbox(frame1, width=200, height=20)
txtout.pack(pady=20)
txtout.grid(row=1, column=1)

app.mainloop()

    
