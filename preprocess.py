import string

def preprocess(text):
    stopwords = {"yang", "di", "ke", "dengan", "dan", "atau", "dari", "untuk", "pada", "adalah", "itu", "ini", "sebuah", "seorang"}
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return [word for word in text.split() if word not in stopwords]
