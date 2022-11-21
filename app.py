
# importing Flask and other modules
from flask import Flask, request, render_template
import pickle
from model import recommendations

# Flask constructor
app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def main():
    return render_template("index.html")

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    input_user = request.form['text_user']
    hasil_recommendation = recommendations(input_user)
    return render_template('prediction.html', tables=[hasil_recommendation.to_html(classes='table table-striped')], titles=hasil_recommendation.columns.values)

@app.route('/about', methods=['GET', 'POST'])
def about():
    return render_template("about.html")

@app.route('/library', methods=['GET', 'POST'])
def library():
    return render_template("library.html")

if __name__=='__main__':
   app.run(debug=True)