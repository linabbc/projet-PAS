from flask import Flask, render_template, request
from Tools import Vectoriser_Question, Calcul_Distance, Get_Question_Reponse

import pandas as pd
import pickle5 as p

app = Flask(__name__)

@app.route('/', methods = ["POST", "GET"])
def index():
    if request.method == "POST":
        merci = True
        with open("NouvellesQuestions.txt", 'a') as writer:
            writer.write(request.form.get("email")+"|"+request.form.get("question")+"\n")
        return render_template("index.html", merci = merci)
    return render_template("index.html")

@app.route('/question', methods = ['POST'])
def question():
    if request.method == 'POST':
        question = request.form.get("InputQuestion")
        with open("corpus.txt", "r") as reader:
            corpus = reader.read()
        corpus = corpus.split("\n")[:-1]
        with open("./Data/Dataset.pkl", "rb") as fh:
            data = p.load(fh)
        question = Vectoriser_Question(question, corpus)
        data = Calcul_Distance(data, question)
        result = Get_Question_Reponse(data)
        return render_template("question.html", questions = result["Question_Origine"].to_list(), responses = result["Reponse"].to_list(), taille = [i for i in range (len(result["Question_Origine"].to_list()))])

@app.route('/formulaire', methods = ['POST'])
def formulaire():
    return render_template("formulaire.html")
        

if __name__=="__main__":
    app.run(debug=True)




