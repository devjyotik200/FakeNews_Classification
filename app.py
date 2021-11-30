import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.models import load_model
import nltk
import re
from nltk.corpus import stopwords
nltk.download('stopwords')

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

# load model
model = load_model('model.h5')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
corpus = []
@app.route('/predict',methods=['POST'])
def predict():
    message=request.form['message']
    data=str(message)
    review=re.sub('[^a-zA-Z]',' ',data)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    onehot_rep = [one_hot(words,5000)for words in corpus]
    embedded_docs = pad_sequences(onehot_rep,padding='pre',maxlen=25)
    test_final = np.array(embedded_docs)
    my_prediction=model.predict_classes(test_final)
    return render_template('result.html',prediction=my_prediction)

    



if __name__ == "__main__":
    app.run(debug=True)