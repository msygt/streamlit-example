import streamlit as st
import numpy as np
import pandas as pd
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import sqlite3

zaman=str(datetime.datetime.now())

conn=sqlite3.connect("trend_yorum.sqlite3")
c=conn.cursor()

c.execute("CREATE TABLE IF NOT EXİST testler(yorum TEXT,sonuc TEXT,zaman TEXT)")
conn.commit()


df=pd.read_csv('trend_yorum.csv.zip',on_bad_lines="skip",delimiter=";")


def temizle(sutun):
    stopwords = ['fakat', 'lakin', 'ancak', 'acaba', 'ama', 'aslında', 'az', 'bazı', 'belki', 'biri', 'birkaç',
                 'birşey', 'biz', 'bu', 'çok', 'çünkü', 'da', 'daha', 'de', 'defa', 'diye', 'eğer', 'en', 'gibi', 'hem',
                 'hep', 'hepsi', 'her', 'hiç', 'için', 'ile', 'ise', 'kez', 'ki', 'kim', 'mı', 'mu', 'mü', 'nasıl',
                 'ne', 'neden', 'nerde', 'nerede', 'nereye', 'niçin', 'niye', 'o', 'sanki', 'şey', 'siz', 'şu', 'tüm',
                 've', 'veya', 'ya', 'yani']
    semboller = string.punctuation
    sutun = sutun.lower()
    for sembol in semboller:
        sutun = sutun.replace(sembol, " ")

    for stopword in stopwords:
        s = " " + stopword + " "
        sutun = sutun.replace(s, " ")

    sutun = sutun.replace("  ", " ")

    return sutun


df['Metin'] = df['Metin'].apply(temizle)

cv=CountVectorizer(max_features=250)
X=cv.fit_transform(df['Metin']).toarray()
y=df['Durum']

x_train,x_test,y_train,y_test=train_test_split(X,y,train_size=0.75,random_state=42)

y=st.text_area("Yorum metni giriniz: ")
btn=st.button("Yorumu kategorilendir")
if btn:
    rf = RandomForestClassifier()
    model = rf.fit(x_train, y_train)
    skor=model.score(x_test, y_test)
    st.balloons()
    tahmin = cv.transform(np.array([y])).toarray()
    kat = {
        1: "Olumlu",
        0: "Olumsuz",
        2: "Nötr"
    }
    sonuc = model.predict(tahmin)
    s=kat.get(sonuc[0])
    st.subheader(s)
    st.write("Model Skoru: ",skor)

    c.execute("INSERT INTO yorumlar,values(?,?,?)",(y,s,zaman))
    conn.commit()

kod="""
import streamlit as st
import numpy as np
import pandas as pd
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df=pd.read_csv('trend_yorum.csv.zip',on_bad_lines="skip",delimiter=";")


def temizle(sutun):
    stopwords = ['fakat', 'lakin', 'ancak', 'acaba', 'ama', 'aslında', 'az', 'bazı', 'belki', 'biri', 'birkaç',
                 'birşey', 'biz', 'bu', 'çok', 'çünkü', 'da', 'daha', 'de', 'defa', 'diye', 'eğer', 'en', 'gibi', 'hem',
                 'hep', 'hepsi', 'her', 'hiç', 'için', 'ile', 'ise', 'kez', 'ki', 'kim', 'mı', 'mu', 'mü', 'nasıl',
                 'ne', 'neden', 'nerde', 'nerede', 'nereye', 'niçin', 'niye', 'o', 'sanki', 'şey', 'siz', 'şu', 'tüm',
                 've', 'veya', 'ya', 'yani']
    semboller = string.punctuation
    sutun = sutun.lower()
    for sembol in semboller:
        sutun = sutun.replace(sembol, " ")

    for stopword in stopwords:
        s = " " + stopword + " "
        sutun = sutun.replace(s, " ")

    sutun = sutun.replace("  ", " ")

    return sutun


df['Metin'] = df['Metin'].apply(temizle)

cv=CountVectorizer(max_features=250)
X=cv.fit_transform(df['Metin']).toarray()
y=df['Durum']

x_train,x_test,y_train,y_test=train_test_split(X,y,train_size=0.75,random_state=42)

y=st.text_area("Yorum metni giriniz: ")
btn=st.button("Yorumu kategorilendir")
if btn:
    rf = RandomForestClassifier()
    model = rf.fit(x_train, y_train)
    skor=model.score(x_test, y_test)
    st.balloons()
    tahmin = cv.transform(np.array([y])).toarray()
    kat = {
        1: "Olumlu",
        0: "Olumsuz",
        2: "Nötr"
    }
    sonuc = model.predict(tahmin)
    s=kat.get(sonuc[0])
    st.subheader(s)
    st.write("Model Skoru: ",skor)
"""

st.code(kod,language="python")
