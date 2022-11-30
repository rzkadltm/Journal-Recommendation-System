from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import pandas as pd
from nltk.tokenize import word_tokenize
import string

data_frame = pd.read_csv('final.csv')

data_frame["title"]=data_frame["Title of the Article"].apply(lambda x: x)
data_frame["abstrak"]=data_frame["Abstract of the article"].apply(lambda x: x)
df = data_frame[['title', 'Authors', 'abstrak']]

df['title'] = df['title'].str.lower() # mengubah menjadi huruf kecil semua
df['abstrak'] = df['abstrak'].str.lower() #df['Authors'] = df['Authors'].str.lower()
df['title'] = df['title'].str.translate(str.maketrans('', '', string.punctuation)) # menghapus kata dari tanda baca
df['abstrak'] = df['abstrak'].str.translate(str.maketrans('', '', string.punctuation)) #df['Authors'] = df['Authors'].str.translate(str.maketrans('', '', string.punctuation))
df['title_abstrak'] = df[['title', 'abstrak']].agg(''.join, axis=1)

cv = CountVectorizer(stop_words='english', tokenizer = word_tokenize)
def recommendations(search):
    
    keywords = cv.fit_transform(df.title_abstrak)
    content = search
    code = cv.transform([content])
    dist = cosine_similarity(code, keywords)
    a = dist.argsort()[0,:-11:-1]
    result = df.loc[a]
    result1 = result[['title', 'Authors', 'abstrak']]
    
    return result1