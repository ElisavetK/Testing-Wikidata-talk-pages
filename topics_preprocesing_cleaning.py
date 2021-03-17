import pandas as pd
import csv

#--------------------------------------------------------
#cut the spaces function
#-----------------------------------------------------------
def remove(string):
    return "".join(string.split())
#--------------------------------------------------------
#RUN AND CREAT THE DATA FRAME. EVERY ROW IS A TOPIC.
#-----------------------------------------------------------

file1 = open('filename_perTopicFiles.txt', 'r')
Lines = file1.readlines()


my_df = pd.DataFrame({'item': pd.Series([], dtype='str'),
                      'text': pd.Series([], dtype='str')})

#create a dataframe with the raw data
c=0
for talk in Lines:
    print (talk)
    filename2=remove(r'perTopicFiles/'+str(talk))
    f=open(str(filename2), 'r', encoding='utf-8')
    my_df.loc[c]=[talk] + [f.read()]
    c+=1

"""
#save a txt. Every raw is a document
my_df_txt=open('WikidataTalkPages.txt', 'w')
for talk in Lines:
    print (talk)
    filename2=remove(r'perTopicFiles/'+str(talk))
    f=open(str(filename2), 'r', encoding='utf-8')
    flines = f.readlines()
    mystr = '\t'.join([fline.strip() for fline in flines])
    print(str(mystr))
    my_df_txt.write(str(mystr) + '\n')
    
my_df_txt.close()
 

f = open("WikidataTalkPages.txt", "r")
print(f.read())

"""

#---------------------------------------------------------------
#REMOVE three ROWS THAT BLOCK THE LANGUAGE DETECTION FUCTION. 
#==================================
#REMOVE ROWS WITH SYMBOLS TO USE langdetect
    
#REMOVE ROWS WITH '--'
df1 = my_df[my_df.text != '--']
#REMOVE ROW WITH https://www.wikidata.org/w/index.php?title=Q8395902&action=history —
df2 = df1[df1.text != 'https://www.wikidata.org/w/index.php?title=Q8395902&action=history —']
#REMOVE ROW WITH '?'
df3 = df2[df2.text != '?']
#===================================================
# REMOVE NON ENGLISH PAGES
    
from langdetect import detect

def langueage_detection(text):
    print(text)
    return detect(text)
  
lag= df3["text"].map(langueage_detection).to_frame()
data_en=df3[(lag['text'] == 'en')]

#--------------------------------------------
#==========================================================
#PREPROCESSING-CLEANING


import nltk
from nltk.tokenize import word_tokenize, regexp_tokenize, wordpunct_tokenize, WhitespaceTokenizer, WordPunctTokenizer
import string
#from nltk.corpus import stopwords
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

#from autocorrect import spell
from calendar import month_name
import re
import enchant
import numpy as np
import enchant.checker
from enchant.checker.CmdLineChecker import CmdLineChecker

#remove urls
def remove_URL(sample):
    """Remove URLs from a sample string"""
    return re.sub(r"http\S+", "", sample)


def remove_non_english_words(token):
    d = enchant.Dict("en_US")
    v=[]
    for i in token:
        v.append(d.check(i))
    return([token[k] for k,val in enumerate(v) if val])
    
    
    

#data_url=data_en["text"].map(remove_URL).to_frame()   
def markup_words_WikidataSymbols(text):
    remove_markup=['user' 'talk', 'span', 'http','utc',
             'int','talkpagelinktext', 'class', 'signaturetalk','ping',
             'quot', 'br','signature', 'font', 'usertalk','wp','de','en', 'wikidata',
             'wikipedia', 'wikimedia']
    return [w for w in text if w not in remove_markup]
 
def extend_stopwords(text):
    remove_markup=['thank','thanks','hi','hello', 'edu', 'couldn','wouldn','hasn', 'haven',
                   'good', 'nice', 'easy', 'easily', 'easier', 'hard', 'harder', 've',
                       'lot', 'right','great','perfect','yeah','yes','no','isn','aren','t']
    return [w for w in text if w not in remove_markup]




text=data_en.loc[14298,'text']

def clean_df(text):
    print(text)
    #remove url
    text0=remove_URL(text)
    
    #remove the phrace 'item documentation'
    if (text0[:22]=='{{item documentation}}') | (text0[:22]=='{{Item documentation}}') | (text0[:22]=='{{item Documentation}}') | (text0[:22]=='{{Item Documentation}}'):
        text1=text0[22:]
    else:  text1=text0
    
    # split into words
    #text_tokenization = word_tokenize(text0)
    #text_regular_expresion=regexp_tokenize(text0,pattern='\w+|\$[\d\.]+|\S+')
    text_wordpunct=wordpunct_tokenize(text1)
    #text_whitespace=WhitespaceTokenizer().tokenize(text0)
    #text_stanford=StanfordTokenizer().tokenize(text0)
    # convert to lower case
    text_lowercase = [w.lower() for w in text_wordpunct]
    
    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    #text_punctuation1= [w.translate(table) for w in text_lowercase]
    text_punctuation= [w for w in text_lowercase if w.translate(table)]
    
    
    # filter out stop words
    #stop_words = set(stopwords.words('english'))
    #text_stopwords = [w for w in text_lowercase if not w in stop_words]
    all_stopwords_gensim = STOPWORDS.union(set(['likes', 'play']))
    text_stopwords=[w for w in text_punctuation if not w in all_stopwords_gensim]
    
    #remove extra stop words
    text_extra_stopwords=extend_stopwords(text_stopwords)
    
    #remove markup 
    text_markup_words=markup_words_WikidataSymbols(text_extra_stopwords)
    
    #remove months
    months = {m.lower() for m in month_name[1:]}  # create a set of month names
    text_no_months=[word for word in text_markup_words if not word in months]
    
    #remove non English words
    #words_engl = set(nltk.corpus.words.words())
    #text_non_english_words=[w for w in text_no_months if w in words_engl or not w.isalpha()]
    
    #replace q with item 
    
    text_item1=['item'  if (val[:1]=='q') and any(chr.isdigit() for chr in val) else val for k,val in enumerate(text_no_months)]
    text_item2=['item'  if val=='q' else val for k,val in enumerate(text_item1)]
    text_item3=[val for k,val in enumerate(text_item2) if not ((val=='item') and (text_item2[(k-1)]=='item')) ]
    
    #replace p with property
    text_property1=['property'  if (val[:1]=='p') and any(chr.isdigit() for chr in val) else val for k,val in enumerate(text_item3)]
    text_property2=['property'  if val=='p' else val for k,val in enumerate(text_property1)]
    text_property3=[val for k,val in enumerate(text_property2) if not ((val=='property') and (text_property2[(k-1)]=='property')) ]
    
    # remove remaining tokens that are not alphabetic
    text_not_alphabetic = [word for word in text_property3 if word.isalpha()]
   
   
    #spelling check
    #spells = [spell(w) for w in (nltk.word_tokenize(text))]
    
    
    #remove sigle letters
    text_single_letters=[w for w in text_not_alphabetic if len(w)>2]
    
    #remove the non Engilish words/lowercase words like april are concidered false
    text_non_english=remove_non_english_words(text_single_letters)
    #remove non English words
    #words_engl = set(nltk.corpus.words.words())
    #text_non_english_words=[w for w in text_no_months if w in words_engl or not w.isalpha()]
    
    #stemming of words
    porter = PorterStemmer()
    text_stemmed = [porter.stem(word) for word in text_non_english]
   
    #lemmatization
    lemmatizer = WordNetLemmatizer()
    text_lemma=[lemmatizer.lemmatize(t, pos="v") for t in text_stemmed]
    
    
    return text_lemma
    

tester= data_en["text"].map(clean_df)
#sa
tester.to_excel("perTopicClean_v4.xlsx", encoding='utf-8')


"""
#================================================================
#LOAD CORRECTLY THE EXCEL
#---------------------------------------------------------
#================================================================
xl = pd.ExcelFile("perTopicClean.xlsx")
dftester = xl.parse('Sheet1')

d1 = dftester['text'].str[1:-1]
d2 = d1.str.replace(r"'", '')
d3= d2.str.replace(r",", '')
d4=pd.DataFrame(d3)
d5=pd.DataFrame({'text': pd.Series([],dtype='str')})
for t in range(len(d4.index)):
    d5.loc[t,'text']=WhitespaceTokenizer().tokenize(d4.loc[t,'text'])
d0=d5.text.tolist()#converts data frames rows to list
#d6=d5.squeeze()#these line converts dataframes to series

"""

import pattern.en as en
base_form = en.lemma('ate')













