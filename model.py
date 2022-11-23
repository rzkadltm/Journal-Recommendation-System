from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import pandas as pd
from nltk.tokenize import word_tokenize

cv = CountVectorizer(stop_words='english', tokenizer = word_tokenize)
data_frame = pd.read_csv('final.csv')

data_frame["Title"]=data_frame["Title of the Article"].apply(lambda x: x)
data_frame["Abstract"]=data_frame["Abstract of the article"].apply(lambda x: x)
df = data_frame[['Title', 'Authors', 'Abstract']]

def recommendations(search):
    
    keyword = cv.fit_transform(df.Title)
    content = search
    code = cv.transform([content])
    dist = cosine_similarity(code, keyword)
    a = dist.argsort()[0, :10]
    result = df.loc[a]
    
    return result