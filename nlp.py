import re
import nltk
import math
from collections import Counter
from itertools import chain

def norm_docs(doc):
    wp = nltk.WordPunctTokenizer()
    stop_words = nltk.corpus.stopwords.words('english')
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)
    doc = doc.lower().strip()
    tokens = wp.tokenize(doc)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    doc = ' '.join(filtered_tokens)
    return doc

def top_n(docs,n=10):
    """ Returns top 10 words from a pandas series"""
    assert docs.__class__.__name__ == 'Series', 'Input is not a pandas series'
    words_counts = Counter(chain.from_iterable([i.split(" ") for i in docs]))
    return sorted(words_counts.items(), key=lambda x: x[1], reverse=True)[:n]


def get_cosine(text1, text2):
    """
    consine similary between two text
    df['vec1']=df['headline'].apply(lambda x: text_to_vector(x)) 
    df['vec2']=df['snippet'].apply(lambda x: text_to_vector(x)) 
    df['simscore']=df.apply(lambda x: get_cosine(x['vec1'],x['vec2']),axis=1)
    """
    w = re.compile(r"\w+")
    vec1 = Counter(w.findall(text1))
    vec2 = Counter(w.findall(text2))
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    return 0.0 if not denominator else float(numerator) / denominator    






