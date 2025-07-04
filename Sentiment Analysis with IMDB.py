from tkinter import *
import customtkinter as c
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def delete():
    txt.delete(0.0, "end")
    
# stopwords are downloaded
nltk.download('stopwords')

# Initialize the TF-IDF vectorizer
cv = TfidfVectorizer(max_features=500)

print("Process Start Please wait")
data = pd.read_csv("IMDB_Dataset_mod.csv")
data.dropna(inplace=True)

print("Data loaded")
# Clean the reviews
def clean(review):
    return " ".join(word for word in review.split() if word.lower() not in stopwords.words('english'))

data['review'] = data['review'].apply(clean)
print("Data cleaned")

# Vectorize the cleaned text
reviews = cv.fit_transform(data['review']).toarray()

# Convert labels to numeric
data['sentiment'] = data['sentiment'].replace(['positive', 'negative'], [1, 0])

# Train,test split
reviews_train, reviews_test, sent_train, sent_test = train_test_split(
    reviews, data['sentiment'], test_size=0.2)

# Train the model
model = LogisticRegression()
model.fit(reviews_train, sent_train)

# Evaluate
predictions = model.predict(reviews_test)
score = accuracy_score(predictions, sent_test)
print(f"Accuracy: {score * 100:.2f}%")

# Predict on new input
def predict_sentiment():
    txtout.delete(0.0,"end")
    text=txt.get(0.0,"end")
    text_cleaned = clean(text)
    text_vector = cv.transform([text_cleaned]).toarray()
    pred = model.predict(text_vector)[0]
    result= "positive review" if pred == 1 else "negative review"
    txtout.insert("0.0",result)


# Using Customtkinter

c.set_appearance_mode('dark')
c.set_default_color_theme('dark-blue')

app = c.CTk()
app.title("text summarization")
app.geometry("400x400")

label1 = c.CTkLabel(app, text="Sentiment Analysis", font=("Arial", 40, "bold"),fg_color="blue")
label1.pack(pady=10)
txt = c.CTkTextbox(app, width=399, height=80)
txt.pack(pady=20)

frame = c.CTkFrame(app)
frame.pack(pady=10)

del_btn = c.CTkButton(frame, text="Clear", command=delete)
sum_btn = c.CTkButton(frame, text="Predict", command=predict_sentiment)
del_btn.grid(row=0, column=0)
sum_btn.grid(row=0, column=1)

frame1=c.CTkFrame(app)
frame1.pack(pady=10)

txtout = c.CTkTextbox(frame1, width=200, height=20)
txtout.pack(pady=20)
txtout.grid(row=1, column=1)

app.mainloop()
print("Process done")
