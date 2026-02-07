import streamlit as st  
import pandas as pd  
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.metrics.pairwise import cosine_similarity  

# ------------------- Custom CSS for Netflix Theme -------------------  
st.markdown("""  
<style>  
    .main { background-color: #141414; color: white; }  
    .stApp { background-color: #141414; }  
    .stTextInput > div > div > input { background-color: #333333; color: white; border: 1px solid #5e0505; border-radius: 4px; padding: 8px; }  
    .stSelectbox > div > div > select { background-color: #333333; color: white; border: 1px solid #5e0505; border-radius: 4px; padding: 8px; }  
    .stRadio > div { background-color: #333333; color: white; border: 1px solid #5e0505; border-radius: 4px; padding: 8px; }  
    .stButton > button { background-color: #5e0505; color: white; border: none; border-radius: 4px; padding: 6px 16px; font-weight: bold; font-size: 14px; transition: background-color 0.3s; }  
    .stButton > button:hover { background-color: #7a0606; }  
    .footer {   
        position: fixed;   
        bottom: 0;   
        left: 0;   
        width: 100%;   
        text-align: center;   
        padding: 10px;   
        background-color: #141414;   
        color: #666;   
        font-size: 12px;   
        border-top: 1px solid #333;   
        z-index: 1000;   
    }  
</style>  
""", unsafe_allow_html=True)  

# ------------------- Load data -------------------  
@st.cache_data  
def load_data():  
    df = pd.read_csv("movies_with_posters.csv")  
    for col in ["title", "genres", "director", "country", "type", "poster_url", "release_year"]:  
        df[col] = df[col].fillna("Unknown")  
    df["title_lower"] = df["title"].str.lower()  
    df["genre_features"] = df["genres"].str.replace(",", " ").str.lower()  
    df["main_genre"] = df["genres"].str.split(",").str[0].str.strip()  

    PLACEHOLDER = "https://via.placeholder.com/120x180?text=No+Image&color=333333&textColor=ffffff"  
    df["poster_url"] = df["poster_url"].replace("", PLACEHOLDER)  
    df["poster_url"] = df["poster_url"].fillna(PLACEHOLDER)  

    df["combined_features"] = (  
        (df["genre_features"] + " ") * 3 +  
        (df["director"] + " ") * 3 +  
        (df["type"] + " ") * 2 +  
        (df["country"] + " ")  
    )  

    content_vectorizer = TfidfVectorizer(stop_words="english")  
    content_matrix = content_vectorizer.fit_transform(df["combined_features"])  
    content_similarity = cosine_similarity(content_matrix)  

    genre_vectorizer = TfidfVectorizer()  
    genre_matrix = genre_vectorizer.fit_transform(df["genre_features"])  
    genre_similarity = cosine_similarity(genre_matrix)  

    return df, content_similarity, genre_similarity  

df, content_similarity, genre_similarity = load_data()  

# ------------------- Recommendation Function -------------------  
def recommend(title, genre, content_type, top_n=10):  
    data = df.copy()  

    if content_type != "Both":  
        data = data[data["type"] == content_type].copy()  

    if genre != "All":  
        data = data[data["genre_features"].str.contains(genre.lower(), case=False, na=False)].copy()  

    title = title.strip().lower()  

    if title and title in data["title_lower"].values:  
        idx = data[data["title_lower"] == title].index[0]  
        filtered_indices = data.index.tolist()  
        filtered_matrix = content_similarity[idx][filtered_indices]  
        scores = list(zip(filtered_indices, filtered_matrix))  
        scores = sorted(scores, key=lambda x: x[1], reverse=True)  
        scores = [s for s in scores if s[0] != idx][:top_n]  
        indices = [i[0] for i in scores]  
        return data.loc[indices].sort_values(by='release_year', ascending=False)  

    elif genre != "All" and not data.empty:  
        base_idx = data.index[0]  
        scores = list(enumerate(genre_similarity[base_idx]))  
        scores = sorted(scores, key=lambda x: x[1], reverse=True)  
        valid_indices = [i for i, score in scores if i in data.index][:top_n]  
        return df.loc[valid_indices].sort_values(by='release_year', ascending=False)  

    return pd.DataFrame()  

# ------------------- Streamlit UI -------------------  
st.set_page_config(page_title="Netflix Recommender", layout="wide")  
st.markdown("<h1 style='text-align: center; color: white;'>Netflix Recommender</h1>", unsafe_allow_html=True)



# Move genre next to search box
col1, col2 = st.columns([3,1])

with col1:
    title_input = st.text_input("Search a Movie or TV Show", placeholder="e.g., The Matrix, Stranger Things")

with col2:
    genre_list = ["All"] + sorted(df["main_genre"].unique())
    genre_input = st.selectbox("Genre", genre_list)

# Keep content type radio below search & genre
type_input = st.radio("Content Type", ["Both", "Movie", "TV Show"], horizontal=True)

# Recommendation button  
if st.button("Recommend", use_container_width=True):  
    results = recommend(title_input, genre_input, type_input)  

    if results.empty:  
        st.warning("No recommendations found. Try another title or genre.")  
    else:  
        num_cols = 5  # keep as original  
        for i in range(0, len(results), num_cols):  
            cols = st.columns(num_cols)  
            batch = results.iloc[i:i+num_cols]  
            for j, (_, row) in enumerate(batch.iterrows()):  
                with cols[j]:  
                    poster = row["poster_url"]  
                    title = row["title"] if row["title"] else "Unknown Title"  
                    year = row["release_year"] if pd.notna(row["release_year"]) else "Unknown"  
                    st.image(poster, width=120)  
                    st.caption(f"{title} ({year})")

# ------------------- Footer -------------------  
st.markdown(  
    """  
    <div class="footer">  
        2025 | developed by Sana Usman  
    </div>  
    """,   
    unsafe_allow_html=True  
)
