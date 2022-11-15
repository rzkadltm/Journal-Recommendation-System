
# importing Flask and other modules
from flask import Flask, request, render_template
import pickle
from model import recommendForSearch

# Flask constructor
app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def main():
    return render_template("index.html")

@app.route('/showtitle', methods=['GET', 'POST'])
def form():
    input_user = request.form['text_user']
    hasil_title = []
    hasil_title = recommendForSearch(input_user).tolist()
    return render_template('predict.html',list=hasil_title)

if __name__=='__main__':
   app.run(debug=True)