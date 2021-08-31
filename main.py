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
df = pd.read_csv("data/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv")


# Keep meaningful columns
clean_df = df[['asins', 'name', 'primaryCategories', 'reviews.rating.1', 'reviews.text']]


# Remove products with few reviews
reviews_by_product = df.groupby(df.name)['reviews.text'].agg('count').reset_index().sort_values(by='reviews.text', ascending=False)
high_reviewed_prods = reviews_by_product[reviews_by_product['reviews.text']>=50]['name'].unique().tolist()
clean_df = clean_df[clean_df['name'].isin(high_reviewed_prods)]


products = ['Amazon Tap Smart Assistant Alexaenabled (black) Brand New',
       'All-New Fire HD 8 Kids Edition Tablet, 8 HD Display, 32 GB, Blue Kid-Proof Case',
       'Amazon Fire HD 8 with Alexa (8" HD Display Tablet)',
       'All-New Fire HD 8 Tablet with Alexa, 8 HD Display, 16 GB, Marine Blue - with Special Offers',
       'All-New Fire HD 8 Kids Edition Tablet, 8 HD Display, 32 GB, Pink Kid-Proof Case',
       'Kindle E-reader - White, 6 Glare-Free Touchscreen Display, Wi-Fi - Includes Special Offers',
       'All-New Fire HD 8 Tablet with Alexa, 8 HD Display, 32 GB, Marine Blue - with Special Offers',
       'Kindle Voyage E-reader, 6 High-Resolution Display (300 ppi) with Adaptive Built-in Light, PagePress Sensors, Wi-Fi - Includes Special Offers',
       'All-New Fire 7 Tablet with Alexa, 7" Display, 8 GB - Marine Blue',
       'Fire Tablet, 7 Display, Wi-Fi, 16 GB - Includes Special Offers, Black',
       'AmazonBasics AAA Performance Alkaline Batteries (36 Count)',
       'Fire HD 10 Tablet, 10.1 HD Display, Wi-Fi, 16 GB - Includes Special Offers, Silver Aluminum',
       'Fire Tablet with Alexa, 7 Display, 16 GB, Blue - with Special Offers',
       'All-New Fire HD 8 Tablet, 8 HD Display, Wi-Fi, 16 GB - Includes Special Offers, Black',
       'Fire Kids Edition Tablet, 7 Display, Wi-Fi, 16 GB, Pink Kid-Proof Case',
       'Kindle Oasis E-reader with Leather Charging Cover - Black, 6 High-Resolution Display (300 ppi), Wi-Fi - Includes Special Offers',
       'Kindle Oasis E-reader with Leather Charging Cover - Merlot, 6 High-Resolution Display (300 ppi), Wi-Fi - Includes Special Offers',
       'Kindle Oasis E-reader with Leather Charging Cover - Walnut, 6 High-Resolution Display (300 ppi), Wi-Fi - Includes Special Offers',
       'AmazonBasics AA Performance Alkaline Batteries (48 Count) - Packaging May Vary',
       'Fire HD 8 Tablet with Alexa, 8 HD Display, 16 GB, Tangerine - with Special Offers',
       'All-New Fire HD 8 Tablet, 8 HD Display, Wi-Fi, 32 GB - Includes Special Offers, Blue',
       'Fire Tablet with Alexa, 7 Display, 16 GB, Magenta - with Special Offers',
       'Fire HD 8 Tablet with Alexa, 8 HD Display, 32 GB, Tangerine - with Special Offers',
       'All-New Fire HD 8 Tablet, 8 HD Display, Wi-Fi, 32 GB - Includes Special Offers, Black',
       'Fire Kids Edition Tablet, 7 Display, Wi-Fi, 16 GB, Blue Kid-Proof Case',
       'Fire Kids Edition Tablet, 7 Display, Wi-Fi, 16 GB, Green Kid-Proof Case',
       'All-New Fire HD 8 Tablet, 8 HD Display, Wi-Fi, 32 GB - Includes Special Offers, Magenta',
       'All-New Fire HD 8 Tablet, 8 HD Display, Wi-Fi, 16 GB - Includes Special Offers, Blue']

# Getting text polarity and subjectivity

clean_df['polarity'] = [TextBlob(review).sentiment.polarity for review in clean_df['reviews.text'].tolist()]
clean_df['subjectivity'] = [TextBlob(review).sentiment.subjectivity for review in clean_df['reviews.text'].tolist()]


# plotting function

def plot_wordclouds(product):
    
    product_reviews = clean_df[clean_df.name == product]
    
    # updating stopwords for non meaningful words
    reviews = str([i for i in product_reviews['reviews.text']])
    words= nltk.tokenize.word_tokenize(reviews)
    reviews_words= [word for word in words if word.isalnum()]
    counter = Counter(reviews_words)
    top_words = [counter.most_common()[i][0] for i in range(0,20)]
    stopwords = STOPWORDS
    my_stop_words = stopwords.update(top_words)
    
    
    # plotting output
    figure, axis = plt.subplots(2, 2, figsize=(20,15))
    figure.suptitle(str('Analysis for: '+product), fontsize=20)
    
    
    #Plotting star review
    total_reviews = pd.DataFrame(product_reviews.groupby(product_reviews['reviews.rating.1'])['reviews.text'].agg('count'))
    labels = total_reviews.index.to_list()
    data = total_reviews['reviews.text']
    
    colors = sns.color_palette('YlOrBr_r')[0:5]
    axis[0,0].pie(data, labels = labels, colors = colors, autopct='%.0f%%')
    axis[0,0].set_title("Number of reviews by star score", fontsize=15)
    axis[0,0].legend()    
    
    #plotting overall sentiment
    axis[0,1].hist(product_reviews.polarity)
    axis[0,1].set_title("Overall review sentiment", fontsize=15)
    
    
    #plotting negative reviews cloud
    negative_reviews = product_reviews[(product_reviews['polarity']<product_reviews['polarity'].quantile(.1))&
                                   (product_reviews['reviews.rating.1'].isin([1,2]))]
    
    negative_reviews_string = str([i for i in negative_reviews['reviews.text']])
    
    #Create negative wordcloud
    cloud = WordCloud(background_color='white', stopwords=my_stop_words).generate(negative_reviews_string)
    
    # Plot negative wordcloud
    axis[1,0].set_title("What negative reviews talk about...", fontsize=15)
    axis[1,0].imshow(cloud, interpolation='bilinear') 
    axis[1,0].axis("off")
    
    #plotting positive reviews cloud
    positive_reviews = product_reviews[(product_reviews['polarity']>product_reviews['polarity'].quantile(.9))&
                                   (product_reviews['reviews.rating.1'].isin([4,5]))]
    positive_reviews_string = str([i for i in positive_reviews['reviews.text']])
    
    #Create positive wordcloud
    cloud = WordCloud(background_color='white', stopwords=my_stop_words).generate(positive_reviews_string)
    
    # Plot positive wordcloud
    axis[1,1].set_title("What positive reviews talk about...", fontsize=15)
    axis[1,1].imshow(cloud, interpolation='bilinear') 
    axis[1,1].axis("off")
    
    plt.show()
    
    return figure
    

st.title('My title')

item = st.selectbox('Select a product to review',products)

st.write(plot_wordclouds(item))