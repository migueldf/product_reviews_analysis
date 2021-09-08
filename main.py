#Library import
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st

import nltk
nltk.download('stopwords')
nltk.download('punkt')

from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS

from gensim.corpora.dictionary import Dictionary

import spacy

from collections import Counter


# Loading datasets
df = pd.read_csv("data/dataset.csv")


products = df.title.unique().tolist()

# plotting function

def plot_wordclouds_dataset(product):
    
    product_reviews = df[df.title == product]
    
    #getting product title
    product = df[df['title']==product].title.unique()[0]
    
    # updating stopwords for non meaningful words
    reviews = str([i for i in product_reviews['content']])
    words= nltk.tokenize.word_tokenize(reviews)
    reviews_words= [word for word in words if word.isalnum()]
    counter = Counter(reviews_words)
    top_words = [counter.most_common()[i][0] for i in range(0,20)]
    stopwords = STOPWORDS
    my_stop_words = stopwords.update(top_words)
    
    
    # plotting output
    figure, axis = plt.subplots(2, 2, figsize=(20,15))
    figure.suptitle(str('Analysis for: '+product), fontsize=20)
    title_font_size = 25
    sns.set_theme(style="whitegrid")
    
    #Plotting star review
    total_reviews = pd.DataFrame(product_reviews.groupby(product_reviews['rating'])['content'].agg('count'))
    labels = total_reviews.index.to_list()
    data = total_reviews['content']
    
    plt.figure(figsize=(15,8))
    colors = sns.color_palette('pastel')[0:5]
    axis[0,0].pie(data, labels = labels, colors = colors, autopct='%.0f%%')
    axis[0,0].set_title("Number of reviews by star score", fontsize=title_font_size)
    axis[0,0].legend()    
    
    #plotting overall sentiment
    pol_sub = pd.DataFrame({'Name':['Comment positivity', 'Comment subjectivity'], 'Values':[np.median(product_reviews.polarity), np.median(product_reviews.subjectivity)]})
    
    sns.set_color_codes("pastel")
    sns.barplot(x="Values", y="Name", data=pol_sub, label="Total", color="b", ax=axis[0,1])
    axis[0,1].set_xticks([-1,0,1])
    axis[0,1].set_xticklabels(['Low','Neutral','High'])
    axis[0,1].set(xlim=(-1, 1), ylabel="",xlabel="")
    axis[0,1].set_title("Overall review sentiment & subjectiveness", fontsize=title_font_size)
    sns.despine(left=True, bottom=True)
    
    
    ######## plotting negative reviews cloud
    
    #getting negative reviews
    negative_reviews = product_reviews[(product_reviews['polarity']<product_reviews['polarity'].quantile(.1))]
    negative_reviews_string = str([i for i in negative_reviews['content']])
    
    negative_blob = TextBlob(negative_reviews_string)
    NounPhrases = negative_blob.noun_phrases
    # Creating an empty list to hold new values
    # combining the noun phrases using underscore to visualize it as wordcloud
    NewNounList=[]
    for words in NounPhrases:
        NewNounList.append(words.replace(" ", "_"))
        
    # Converting list into a string to plot wordcloud
    NegativeNewNounString=' '.join(NewNounList)
    negative_cloud = WordCloud(stopwords=my_stop_words, background_color='white', colormap='Reds').generate(NegativeNewNounString)
    
    ######## plotting positive reviews cloud
    
    #
    positive_reviews = product_reviews[(product_reviews['polarity']>product_reviews['polarity'].quantile(.9))]
    positive_reviews_string = str([i for i in positive_reviews['content']])
    
    positive_blob = TextBlob(positive_reviews_string)
    NounPhrases = positive_blob.noun_phrases
    # Creating an empty list to hold new values
    # combining the noun phrases using underscore to visualize it as wordcloud
    NewNounList=[]
    for words in NounPhrases:
        NewNounList.append(words.replace(" ", "_"))
        
    # Converting list into a string to plot wordcloud
    PositiveNewNounString=' '.join(NewNounList)
    
    positive_cloud = WordCloud(stopwords=my_stop_words, background_color='white', colormap='Greens_r').generate(PositiveNewNounString)
    
    # Plot negative wordcloud
    axis[1,0].set_title("What negative reviews talk about...", fontsize=title_font_size)
    axis[1,0].imshow(negative_cloud, interpolation='bilinear')
    axis[1,0].axis("off")
    
    # Plot positive wordcloud
    axis[1,1].set_title("What positive reviews talk about...", fontsize=title_font_size)
    axis[1,1].imshow(positive_cloud, interpolation='bilinear') 
    axis[1,1].axis("off")
    
    plt.show()
    
    return figure
    

st.title('🌳🐒 Product Jungle Gibber Analizer 🐒🌳')

st.write("On 1995, Amazon.com, Inc. started letting customer post both negative and positive reviews to the products they bought. This move was considered nuts by other retailers, but Amazon vision went far beyond sales: they understood the valuer of earning people's trust by enabling to make smart and informed purchases.\n\n\n This project aims to build towards this purpuse by enabling customers to get a brief overview of customer's feelings related to products.\n\n The below dashboard is formed by 3 components:\n\n\n 1) Star rating distribution. \n\n 2) Reviews positiveness & subjectivity: This section evaluates customer sentiments through machine learning techniques to get an overall idea of customer's opinions regarding a product.\n\n 3) Recurrent positive & Negative comments: Know what people are saying regarding a product.")

item = st.selectbox('Start by selecting a product to review:',products)

st.write(plot_wordclouds_dataset(item))

