from flask import Flask,request,jsonify, render_template
import joblib
import numpy as np 

app = Flask(__name__)
model = joblib.load("model.joblib")
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods = ['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_feature = [np.array(features)]
    prediction = model.predict(final_feature)
    output = prediction[0]
    return render_template('index.html',prediction_text =f'predicted class:{output}')


if __name__ == '__main__':
    app.run()

    
 