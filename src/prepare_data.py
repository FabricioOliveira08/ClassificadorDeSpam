import pandas as pd

df = pd.read_csv("data/SMSSpamCollection.txt", sep="\t", header=None)

df.columns = ["label", "message"]

df.to_csv("data/spam.csv", index=False)