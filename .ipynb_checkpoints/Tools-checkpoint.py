#!/usr/bin/env python
# coding: utf-8

# # Tools
# 
# *Ce notebook sert principalement à définir des fonctions et outils que nous allons utiliser au long de ce projet*
# 
# **le 20/11/21**
# 
# ---

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from unidecode import unidecode
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import FrenchStemmer
from nltk.corpus import stopwords
from nltk.metrics import *

def preprocessing(questions):
    """Retourne un dataframe où les questions ont subies les différentes étapes de préprocessing:
    
    - Suppression des majuscules.
    
    - Suppression des accents.
    
    - Tokenisation ( à l'aide de la regex suivante : [a-zA-Z]+ seules les mots sont conservés).
    
    - Suppression des stops words en français.
    
    - Stemmatisation 
    
    questions -- Série pandas comprenant les questions que l'on souhaite utiliser comme référence
    pour le chatbot.
    """
    
    # Nous commençons par retirer les accents et les majuscules sur les mots.
    questions = questions.apply(lambda sentence : sentence.lower())
    questions = questions.apply(lambda sentence : unidecode(sentence))
    # Pour les questions, nous ne gardons que les mots, nous supprimons tous le reste.
    tokenizer = RegexpTokenizer("[a-zA-Z]+")
    questions = questions.apply(lambda sentence : tokenizer.tokenize(sentence))
    # On retire les stops words.
    stops = set(stopwords.words("french"))
    questions = questions.apply(lambda words : [word for word in words if word not in stops and len(word)>2])
    # Nous lemmatisons les mots.
    stemmer = FrenchStemmer()
    questions = questions.apply(lambda words : [stemmer.stem(word) for word in words])
    #lemmer = WordNetLemmatizer()
    #questions = questions.apply(lambda words : [lemmer.lemmatize(word) for word in words])
    return questions

def Vectoriser(tool, data, corpus, to_transform = None):
    """Retourne du texte vectorisé
    
    tool -- Un Vectorizer.
    
    data -- Le dataframe auquel nous rattachons les résultats.
    
    corpus -- La liste des phrases d'où nous allons piocher les mots
    pour la vectorisation.
    
    to_transform -- la liste des phrases à vectorizer.
    
    Si to_transform n'est pas rempli, c'est que l'on souhaite rajouter de 
    nouvelles questions à la liste des questions que l'on possède déjà.
    """
    if to_transform is None:
        X = tool.fit_transform(corpus)
        vectorised = [list(elem) for elem in X.toarray()]
        column = str(tool)[:-2]
    else:
        tool.fit(corpus)
        X = tool.transform(to_transform)
        vectorised = [list(elem) for elem in X.toarray()]
        column = str(tool)[:-2]
    return pd.concat([data, pd.DataFrame({column : vectorised})], axis = 1)

def Vectoriser_Question(question, corpus):
    """Retourne la question qui est posée par l'utilisateur sous forme d'un dataframe
    comprenant la question vectorisée de deux manières différentes, à l'aide d'un CountVectorizer
    et d'un TfidfVectorizer.
    
    question -- La question posée par l'utilisateur
    
    corpus -- La liste des phrases d'où nous allons piocher les mots
    pour la vectorisation. 
    
    """
    question = pd.DataFrame({"Question_posee" : [question]})
    question = preprocessing(question["Question_posee"])
    to_transform = [" ".join(question[0])]
    question = Vectoriser(CountVectorizer(), question, corpus, to_transform)
    question = Vectoriser(TfidfVectorizer(), question, corpus, to_transform)
    return question

def Calcul_Distance(data, question):
    """Renvoie un dataset auquel on ajoute 4 colonnes:
    
    Cosinus_CountVectorizer : Cosinus entre le paramètre 'question' et les questions dont nous possedons les réponses qui ont été vectorisé à l'aide d'un CountVectorizer.
    
    Cosinus_TfidfVectorizer : Cosinus entre le paramètre 'question' et les questions dont nous possedons les réponses qui ont été vectorisé à l'aide d'un TfidfVectorizer.
    
    Minkowski_CountVectorizer : Distance entre le paramètre 'question' et les questions dont nous possedons les réponses qui ont été vectorisé à l'aide d'un CountVectorizer.
    (Avec p = 0.1)
    
    Monkowski_TfidfVectorizer : Distance entre le paramètre 'question' et les questions dont nous possedons les réponses qui ont été vectorisé à l'aide d'un TfidfVectorizer.
    (Avec p = 0.1)
    
    data -- DataFrame contenant les questions dont on connait les réponses. Celle-ci devant être vectorisées.
    
    question -- DataFrame contenant la question vectorisée.
    """
    data["Cosinus_CountVectorizer"] = [np.dot(vector, question["CountVectorizer"][0])/(np.linalg.norm(vector)*np.linalg.norm(question["CountVectorizer"][0])) for vector in data["CountVectorizer"]]
    data["Cosinus_TfidfVectorizer"] = [np.dot(vector, question["TfidfVectorizer"][0])/(np.linalg.norm(vector)*np.linalg.norm(question["TfidfVectorizer"][0])) for vector in data["TfidfVectorizer"]]
    data["Minkowski_CountVectorizer"] = [np.linalg.norm(np.array(vector)-np.array(question["CountVectorizer"][0]), ord=0.1) for vector in data["CountVectorizer"]]
    data["Minkowski_TfidfVectorizer"] = [np.linalg.norm(np.array(vector)-np.array(question["TfidfVectorizer"][0]), ord=0.1) for vector in data["TfidfVectorizer"]]
    return data

def Get_Question_Reponse(data):
    """Renvoie un DataFrame contenant les questions et réponses les plus problables compte tenu de la question posée.
    
    data -- DataFrame contenant les questions dont on connait les réponses. Celle-ci devant être vectorisées.
    
    """
    
    #Pour les distance basées sur un calcul d'angle, on prend la valeur la plus grande car plus proche de 1.
    response1 = data[data["Cosinus_CountVectorizer"] == max(data["Cosinus_CountVectorizer"])][["Question_Origine", "Reponse"]]
    response2 = data[data["Cosinus_TfidfVectorizer"] == max(data["Cosinus_TfidfVectorizer"])][["Question_Origine", "Reponse"]]
    
    #Pour les distances basées sur un calcul type Minkowski, on prend le minimum car nous cherchons les vecteurs les plus proches (les moins loins).
    response3 = data[data["Minkowski_CountVectorizer"] == min(data["Minkowski_CountVectorizer"])][["Question_Origine", "Reponse"]]
    response4 = data[data["Minkowski_TfidfVectorizer"] == min(data["Minkowski_TfidfVectorizer"])][["Question_Origine", "Reponse"]]
    return pd.concat([response1, response2, response3, response4]).drop_duplicates()