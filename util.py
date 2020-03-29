
'''
   utility functions for processing terms

    shared by both indexing and query processing
'''

from nltk.stem import PorterStemmer, SnowballStemmer
import string
import math
from metrics import logarithmic_avg, probabilistic_idf, augmented_tf
import time

# globals
__STOPWORDS__ = {}
__PUNCTUATIONS__ = string.punctuation+"".join(["...", ".........." , "....","-", "--","/"])
CONTRACTION_MAP = {
"ain't": "is not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I would",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}


with open('stopwords', "r") as fd:
    print("stopwords read!")
    # create a lookup DS
    __STOPWORDS__ = fd.read().split()

# utils
def isStopWord(word):
    ''' using the NLTK functions, return true/false'''
    # some stopwords specific to the corpus
    return word in __STOPWORDS__ or word in ["dash"]


def punctuation(word):
    return word in __PUNCTUATIONS__

def stemming(word):
    #__stemmer__ = SnowballStemmer(language="english")
    __stemmer__ = PorterStemmer()
    return __stemmer__.stem(word)
    ''' return the stem, using a NLTK stemmer. check the project description for installing and using it'''


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print ('Execution of %r took %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed

def get_norm(vec):
    # root over x^2+ y^2 + Z^2
    return math.sqrt(sum([a*a for a in vec]))

def dot(vec_a, vec_b):
    return sum([x*y for x,y in zip(vec_a, vec_b)])

def cosine_similarity(x,y):
    norm_x = get_norm(x)
    norm_y = get_norm(y)
    dot_x_y = dot(x,y)
    if norm_x == 0 or norm_y == 0:
        return 0.0
    return round(dot_x_y/(norm_x * norm_y),5)



CLASS_MAPPINGS = {"comp.graphics": 1,
                      "comp.os.ms-windows.misc": 1,
                      "comp.sys.ibm.pc.hardware": 1,
                      "comp.sys.mac.hardware": 1,
                      "comp.windows.x": 1,
                      "rec.autos": 2,
                      "rec.motorcycles": 2,
                      "rec.sport.baseball": 2,
                      "rec.sport.hockey": 2,
                      "sci.crypt": 3,
                      "sci.electronics": 3,
                      "sci.med": 3,
                      "sci.space": 3,
                      "misc.forsale": 4,
                      "talk.politics.misc": 5,
                      "talk.politics.guns": 5,
                      "talk.politics.mideast": 5,
                      "talk.religion.misc": 6,
                      "alt.atheism": 6,
                      "soc.religion.christian": 6}


def argmax(array):
    max = -99999
    max_idx = 0
    for idx, item in enumerate(array):
        if item > max:
            max = item
            max_idx = idx
    return max_idx




# def log_tf(fn):
#     def wrapper(*args):
#         return fn(*args)
#     return wrapper
#
# def augment_tf(fn):
#     def wrapper(*args):
#
#         return fn(*args)
#     return wrapper
#
# def
#
#
# def prob_idf(fn):
#     def wrapper(*args):
#         return fn(*args)
#     return wrapper
#
#
# if __name__ == "__main__":
#     @log_tf
#     def dummy(self, tf, idf):
#         print(self, tf.__name__, idf.__name__)
#
#     dummy(None, augmented_tf, probabilistic_idf)




