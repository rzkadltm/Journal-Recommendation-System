from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import pandas as pd
from nltk.tokenize import word_tokenize
import string
import nltk
nltk.download('punkt')
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import pandas as pd
from nltk.tokenize import word_tokenize
import string
import warnings
warnings.filterwarnings("ignore")

data_frame = pd.read_csv('final.csv')
data_frame.tail()

data_frame["title"]=data_frame["Title of the Article"].apply(lambda x: x)
data_frame["abstrak"]=data_frame["Abstract of the article"].apply(lambda x: x)

data = data_frame[['title', 'Authors', 'abstrak']]
data['title_abstrak'] = data[['title', 'abstrak']].agg(''.join, axis=1)
df = data.dropna()
df

# mengubah menjadi huruf kecil semua
df['title'] = df['title'].str.lower()
df['title_abstrak'] = df['title_abstrak'].str.lower()
df['abstrak'] = df['abstrak'].str.lower()

# menghapus kata dari tanda baca
df['title'] = df['title'].str.translate(str.maketrans('', '', string.punctuation))
df['title_abstrak'] = df['title_abstrak'].str.translate(str.maketrans('', '', string.punctuation))
df['abstrak'] = df['abstrak'].str.translate(str.maketrans('', '', string.punctuation))


cv = CountVectorizer(stop_words='english', tokenizer = word_tokenize)
def recommendations(search):
    
    keywords = cv.fit_transform(df.title_abstrak)
    contents = search
    content = contents.translate(str.maketrans('', '', string.punctuation))
    content1 = content.lower()
    code = cv.transform([content1])
    dist = cosine_similarity(code, keywords)
    a = dist.argsort()[0,:-11:-1]
    result = df.loc[a]
    result1 = result[['title', 'Authors', 'abstrak']]
    
    return result1
    
recommendations('Machine learning for engineering')
