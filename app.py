import streamlit as st
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pyngrok import ngrok

# Load and preprocess the dataset
df = pd.read_csv("0_news_articles.csv")

nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [ps.stem(word) for word in text if word.isalnum() and word not in stop_words]
    return " ".join(text)

df['processed_content'] = df['Description'].apply(preprocess_text)

# Vectorize the content
tfidf = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['processed_content'])

# Compute similarity matrix
similarity_matrix = cosine_similarity(tfidf_matrix)

# Define recommendation function
def recommend_news(article_index, num_recommendations=5):
    similarity_scores = list(enumerate(similarity_matrix[article_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similar_articles = similarity_scores[1:num_recommendations+1]
    recommended_indices = [i[0] for i in similar_articles]
    
    # Return the title, content, date, category, and URL
    return df.iloc[recommended_indices][['Title', 'Description', 'Date', 'Category', 'URL']]

# Enhanced Streamlit UI Design
st.set_page_config(page_title="News Aggregator", layout="wide")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📰 News Aggregation System</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Content-Based Filtering for News Recommendations</h2>", unsafe_allow_html=True)

# Create a layout with two columns: one for article selection, one for displaying results
col1, col2 = st.columns(2)

# News article selection in the first column
with col1:
    st.subheader("Select an Article")
    article_titles = df['Title'].tolist()
    selected_title = st.selectbox("Choose a news article to get recommendations:", article_titles)
    
    # Display the selected article's content
    article_index = df[df['Title'] == selected_title].index[0]
    st.write("### Selected Article")
    st.write(f"{df.iloc[article_index]['Title']}")
    st.write(f"Date: {df.iloc[article_index]['Date']}")
    st.write(f"Category: {df.iloc[article_index]['Category']}")
    st.write(f"URL: {df.iloc[article_index]['URL']}")
    st.write(df.iloc[article_index]['Description'])

# Recommendations in the second column
with col2:
    st.subheader("Recommended Articles")
    recommended_articles = recommend_news(article_index)
    
    for index, row in recommended_articles.iterrows():
        st.write(f"Title: {row['Title']}")
        st.write(f"Date: {row['Date']}")
        st.write(f"Category: {row['Category']}")
        st.write(f"URL: {row['URL']}")
        st.write(row['Description'])
        st.write("---")

# Footer
st.markdown(
    """
    <style>
        footer {visibility: hidden;}
        .footer-text {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background: linear-gradient(to right, #00c6ff, #0072ff);
            color: white;
            text-align: center;
            padding: 20px;
            font-family: 'Arial', sans-serif;
            font-size: 16px;
            box-shadow: 0px -3px 5px rgba(0,0,0,0.2);
        }
        .footer-text p {
            margin: 0;
            padding: 0;
        }
        .footer-text a {
            color: #ffcc00;
            text-decoration: none;
            font-weight: bold;
        }
        .footer-text a:hover {
            color: #fff;
            text-decoration: underline;
        }
    </style>
    <div class="footer-text">
        <p>Powered by <a href="https://streamlit.io" target="_blank">Streamlit</a> | Content-Based Filtering System</p>
    </div>
    """, unsafe_allow_html=True
)



# Start ngrok tunnel on port 8501
public_url = ngrok.connect(8501, "http")
print("Streamlit App URL:", public_url)


