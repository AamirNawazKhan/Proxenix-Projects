from tkinter import *
import customtkinter as c
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter
from heapq import nlargest

spacy.cli.download("en_core_web_lg")
nlp = spacy.load("en_core_web_lg")

def delete():
    txt.delete(0.0, "end")


def summary():
    text=txt.get("0.0","end")
    doc = nlp(text)

    tokens = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct and token.text != "\n"]

    # print(tokens)

    tokens1 = []
    stopwords = list(STOP_WORDS)
    allowed_pos = ["ADJ", "PROPN", "VERB", "NOUN"]
    for token in doc:
        if token.text in stopwords or token.text in punctuation:
            continue
        if token.pos_ in allowed_pos:
            tokens1.append(token.text)

    # print(tokens1)

    words_freq = Counter(tokens)

    max_freq = max(words_freq.values())

    # print(max_freq)

    for word in words_freq.keys():
        words_freq[word] = words_freq[word] / max_freq

    # print(words_freq)

    sent_token = [sent.text for sent in doc.sents]

    # print(sent_token)

    sent_score = {}
    for sent in sent_token:
        for word in sent.split():
            if word.lower() in words_freq.keys():
                if sent not in sent_score.keys():
                    sent_score[sent] = words_freq[word]
                else:
                    sent_score[sent] += words_freq[word]

    # print(sent_score)

    # print(pd.DataFrame(list(sent_score.items()),columns=["Sentences","Score"]))

    num_sentences = 3
    a = nlargest(num_sentences, sent_score, key=sent_score.get)
    b = " ".join(a)

    txtout.insert(0.0, b)


#Made by Aamir Nawaz Khan

c.set_appearance_mode('dark')
c.set_default_color_theme('dark-blue')

app = c.CTk()
app.title("text summarization")
app.geometry("400x400")

label1 = c.CTkLabel(app, text="text summarization", font=("Arial", 20, "bold"))
label1.pack(pady=10)
txt = c.CTkTextbox(app, width=1000, height=300)
txt.pack(pady=20)

frame = c.CTkFrame(app)
frame.pack(pady=10)

del_btn = c.CTkButton(frame, text="Clear", command=delete)
sum_btn = c.CTkButton(frame, text="Summarize", command=summary)
del_btn.grid(row=0, column=0)
sum_btn.grid(row=0, column=1)

n = ['3', '4', '5']
combox = c.CTkComboBox(frame, values=n, state="readonly")
combox.set('Select no of lines')
combox.grid(row=1, column=0)
no = combox.get()

frame1=c.CTkFrame(app)
frame1.pack(pady=10)

txtout = c.CTkTextbox(frame1, width=1000, height=300)
txtout.pack(pady=20)
txtout.grid(row=1, column=1)

app.mainloop()
