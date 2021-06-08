  
from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle

# load the model from disk

filename = 'nlp_model.pkl'
clf = pickle.load(open(filename,'rb'))

# load count vectorizer
cv =  pickle.load(open('transform.pkl','rb'))

# Initialize flask

app = Flask(__name__)


# Here it invoked by  the webpage to pass its input user data
@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods =['POST'])
def predict():
    
    if request.method == 'POST':
        message = request.form['message']  ## Here message col should be same name as used in home.html
        data = [message]
        vect = cv.transform(data).toarray()  ## Apply Countvector Pickle file to convert and transfered to array
        my_prediction = clf.predict(vect)
        
        ## Here prediction is the output col same name used in result.html
    return render_template('result.html',prediction = my_prediction)
## Here 2 html file used one for input read(home.html) and another to output /show result (result.html)


if __name__ == '__main__':
	app.run(debug=True)