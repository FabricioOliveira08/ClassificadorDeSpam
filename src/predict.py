import joblib

model = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

mensagem = input("Digite uma mensagem: ")

X = vectorizer.transform([mensagem])    

resultado = model.predict(X)

if(resultado[0] == "spam"):
    print("Spam")
else:
    print("Não Spam")
