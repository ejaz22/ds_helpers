#2. process text - white space, special character etc
import re
import nltk

def norm_docs(doc):
    wp = nltk.WordPunctTokenizer()
    stop_words = nltk.corpus.stopwords.words('english')
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)
    doc = doc.lower().strip()
    tokens = wp.tokenize(doc)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    doc = ' '.join(filtered_tokens)
    return doc

# 2. get consine similary between two text
import math
import re
from collections import Counter

# get cosine similarity
def get_cosine(text1, text2):
    w = re.compile(r"\w+")
    vec1 = Counter(w.findall(text1))
    vec2 = Counter(w.findall(text2))
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    return 0.0 if not denominator else float(numerator) / denominator

    
   
 # cosine similary to compare two documents
import math
import re
from collections import Counter
import pandas as pd

WORD = re.compile(r"\w+")


def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)

df=pd.read_csv('/content/drive/article.csv')
df['vector1']=df['headline'].apply(lambda x: text_to_vector(x)) 
df['vector2']=df['snippet'].apply(lambda x: text_to_vector(x)) 
df['simscore']=df.apply(lambda x: get_cosine(x['vector1'],x['vector2']),axis=1)
