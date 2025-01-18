from flask import Flask,render_template,request
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
import string

app = Flask(__name__)
nltk.download('punkt_tab')
nltk.download('stopwords')
ps = PorterStemmer()

def transform_text(text:str)->str:
    text = text.lower()
    ## tokenizing the text 
    text = nltk.word_tokenize(text)
    ## removing words which are not alnum
    y = []
    for word in text:
        if word.isalnum():
            y.append(word)

    text = y[:]
    y.clear()
    for word in text:
        if word not in stopwords.words('english') and word not in string.punctuation:
            y.append(word)
    text = y[:]
    y.clear()
    for word in text:
        y.append(ps.stem(word))
    return ' '.join(y).strip()
    

def predict_spam(text:str):
    vectorized_text = tfidf.transform([text])
    prediction = model.predict(vectorized_text)[0]
    return prediction


@app.route("/predict",methods=["POST"])
def predict():
    if request.method=='POST':
        input_text = request.form['message']
        transformed_text = transform_text(input_text)
        result = predict_spam(transformed_text)
        return render_template('index.html',result=result)

@app.route('/')
def home():
    return render_template('index.html')


if __name__== "__main__":
    model = pickle.load(open("model.pkl",'rb'))
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    app.run(host="0.0.0.0")