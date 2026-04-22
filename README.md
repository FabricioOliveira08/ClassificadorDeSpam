# Classificador de Spam com Machine Learning

## Sobre o Projeto

Este projeto utiliza técnicas de Machine Learning para identificar automaticamente se uma mensagem de texto é spam ou não.

O sistema foi desenvolvido em Python utilizando:

* Pandas
* Scikit-learn
* Joblib
* Streamlit

O modelo foi treinado com o dataset **SMS Spam Collection**, utilizando:

* Vetorização de texto com `TfidfVectorizer`
* Classificação com `Naive Bayes`

Além do treinamento do modelo, o projeto também possui:

* Persistência do modelo treinado
* Sistema de predição
* Interface web interativa com Streamlit

---

# Tecnologias Utilizadas

* Python 3
* Pandas
* Scikit-learn
* Joblib
* Streamlit

---

# Estrutura do Projeto

```bash
ClassificadorDeSpam/
│
├── data/
│   ├── SMSSpamCollection.txt
│   └── spam.csv
│
├── model/
│   ├── spam_model.pkl
│   └── vectorizer.pkl
│
├── src/
│   ├── prepare_data.py
│   ├── train.py
│   └── predict.py
│
├── venv/
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Dataset

O projeto utiliza o dataset:

**SMS Spam Collection**

O dataset contém mensagens classificadas em duas categorias:

* `spam` → mensagens indesejadas
* `ham` → mensagens normais

---

# Etapas do Projeto

## 1. Preparação dos Dados

O dataset original foi convertido para um arquivo CSV contendo:

* `label` → classificação da mensagem
* `message` → conteúdo textual

Exemplo:

| label | message                       |
| ----- | ----------------------------- |
| ham   | Oi, tudo bem?                 |
| spam  | WINNER!! Claim your prize now |

---

## 2. Treinamento do Modelo

O treinamento é realizado no arquivo:

```bash
src/train.py
```

Etapas executadas:

* Leitura dos dados
* Separação entre treino e teste
* Vetorização do texto
* Treinamento do modelo
* Avaliação da acurácia
* Salvamento do modelo

---

## 3. Vetorização de Texto

Foi utilizado:

```python
TfidfVectorizer(ngram_range=(1,2))
```

O modelo aprende:

* palavras individuais
* combinações de palavras

Exemplo:

```text
free
money
free money
```

---

## 4. Modelo Utilizado

O algoritmo utilizado foi:

```python
MultinomialNB()
```

---

## 5. Persistência do Modelo

Após o treinamento, o modelo e o vectorizer são salvos utilizando `joblib`:

```python
joblib.dump(model, "model/spam_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")
```

Isso evita a necessidade de treinar o modelo novamente a cada execução.

---

## 6. Sistema de Predição

O arquivo:

```bash
src/predict.py
```

permite:

* carregar o modelo salvo
* receber mensagens do usuário
* classificar novas mensagens

Exemplo:

```text
Digite uma mensagem:
WINNER!! You have won a prize

Resultado:
Spam
```

---

# Interface Web

A interface foi desenvolvida com Streamlit.

Arquivo principal:

```bash
app.py
```

Funcionalidades:

* Campo de texto
* Botão de classificação
* Exibição visual do resultado
* Confiança do modelo

---

# Como Executar o Projeto

## 1. Clonar o repositório

```bash
git clone <URL_DO_REPOSITORIO>
```

---

## 2. Criar ambiente virtual

```bash
python -m venv venv
```

---

## 3. Ativar ambiente virtual

### Windows

```bash
venv\Scripts\activate
```

---

## 4. Instalar dependências

```bash
pip install -r requirements.txt
```

---

## 5. Treinar o modelo

```bash
python src/train.py
```

---

## 6. Executar interface web

```bash
streamlit run app.py
```

---

# Resultados

O modelo alcançou aproximadamente:

```text
99% de acurácia
```

---

# Limitações

O modelo utiliza técnicas simples de NLP.

Apesar da alta acurácia, ele pode apresentar dificuldades em mensagens muito diferentes do padrão presente no dataset.

Isso ocorre porque o modelo aprende padrões de frequência de palavras, não entendimento semântico profundo.

---

# Possíveis Melhorias

* Adicionar histórico de mensagens
* Melhorar o design da interface
* Implementar deploy online
* Testar modelos mais avançados
* Utilizar redes neurais ou transformers

---

Projeto desenvolvido para fins de estudo e prática em:

* Machine Learning
* NLP
* Python
* Streamlit

