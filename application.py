import tkinter as tk
# from application import tfidf, tfidf_matrix, recommendations_text, movies_entry
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
from data_processing import netflix_data
from sklearn.feature_extraction.text import TfidfVectorizer

# Define function to generate recommendations
def generate_recommendations():
    # Process user input
    user_input = movies_entry.get().strip()

    # Perform fuzzy matching to find similar movie titles
    matches = process.extract(user_input, netflix_data['title'], limit=5)
    matched_indices = [netflix_data[netflix_data['title'] == match[0]].index[0] for match in matches]

    # Extract additional features
    selected_movies = netflix_data.iloc[matched_indices]
    selected_features = selected_movies['genres'] + ' ' + \
                        selected_movies['production_companies'].astype(str) + ' ' + \
                        selected_movies['release_date'].astype(str) + ' ' + \
                        selected_movies['popularity'].astype(str)

    # Calculate cosine similarity with matched movies and additional features
    selected_tfidf_matrix = tfidf.transform(selected_features)
    selected_cosine_sim = cosine_similarity(selected_tfidf_matrix, tfidf_matrix).mean(axis=0)

    # Get indices of top recommendations
    top_indices = selected_cosine_sim.argsort()[-11:-1][::-1]  # Recommend top 10 movies
    recommended_movies = netflix_data.iloc[top_indices]

    # Display recommendations
    recommendations_text.delete(1.0, tk.END)
    recommendations_text.insert(tk.END, "Recommended Movies:\n")
    for _, movie in recommended_movies.iterrows():
        recommendations_text.insert(tk.END, movie['title'] + "\n")


# Create GUI
root = tk.Tk()
root.title("Movie Recommendation System")

# GUI components
movies_label = tk.Label(root, text="Enter Movie Name:")
movies_label.grid(row=0, column=0)
movies_entry = tk.Entry(root)
movies_entry.grid(row=0, column=1)

recommend_button = tk.Button(root, text="Get Recommendations", command=generate_recommendations)
recommend_button.grid(row=1, columnspan=2)

recommendations_text = tk.Text(root, height=15, width=50)
recommendations_text.grid(row=2, columnspan=2)

# Load and preprocess data
tfidf = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2), min_df=5)
tfidf_matrix = tfidf.fit_transform(netflix_data['genres'] + ' ' + \
                                   netflix_data['production_companies'].astype(str) + ' ' + \
                                   netflix_data['release_date'].astype(str) + ' ' + \
                                   netflix_data['popularity'].astype(str))

root.mainloop()
