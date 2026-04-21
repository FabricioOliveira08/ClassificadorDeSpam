import streamlit as st
import joblib


model = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

st.title("Classificador de Spam")

mensagem = st.text_input("Digite uma mensagem: ")

if st.button("Classificar"):
    X = vectorizer.transform([mensagem])        
    resultado = model.predict(X)

    st.subheader("Resultado: ")
    if(resultado[0] == "spam"):
        st.write("Spam")
    else:
        st.write("Não Spam")
