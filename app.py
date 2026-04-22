import streamlit as st
import joblib

st.set_page_config(
    page_title="Classificador de Spam",
    layout="centered"
)

model = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

st.title("Classificador de Spam")
st.markdown("Digite uma mensagem abaixo e descubra se ela é **Spam ou Não Spam** usando Machine Learning.")

st.divider()

mensagem = st.text_input("Digite uma mensagem: ")

if st.button("Classificar"):

    if(mensagem.strip() == ""):
        st.warning("⚠️ Digite uma mensagem antes de classificar.")
    else:
        X = vectorizer.transform([mensagem])        
        resultado = model.predict(X)
        prob = model.predict_proba(X)

        st.divider()
        st.subheader("Resultado: ")

        if(resultado[0] == "spam"):
            st.error("Spam detectado")
        else:
            st.success("A mensagem não é Spam")
        
        confianca = prob.max()
        st.write(f" Confiança do modelo: **{confianca:.2%}**")