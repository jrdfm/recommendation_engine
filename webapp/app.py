import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from typing import Optional # Import Optional

# --- Configuration & Data Loading --- 
DATA_PATH = "../data/content_raw.csv" # Path relative to app.py
MAX_FEATURES = 5000 # For TF-IDF Vectorizer

def load_and_prepare_data(path):
    """Loads data, performs basic cleaning and TF-IDF calculation."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at: {path}. Run the data fetching script first.")

    print(f"Loading data from {path}...")
    df = pd.read_csv(path)

    # --- Basic Preprocessing (simplified version from notebook) ---
    # Drop potential duplicates based on id and type
    df = df.drop_duplicates(subset=['id', 'type'], keep='first')
    # Drop rows with critical missing info
    df.dropna(subset=['overview', 'genre_names', 'title'], inplace=True)
    # Convert columns safely
    for col in ['overview', 'genre_names', 'title', 'poster_path']: # Added poster_path
        if col in df.columns:
             df[col] = df[col].astype(str).fillna('') # Ensure string and fillna

    # Combine features for TF-IDF
    df['tags'] = df['overview'].str.lower() + ' ' + \
                   df['genre_names'].str.replace(',', ' ').str.lower() + ' ' + \
                   df['title'].str.lower()
    df['tags'] = df['tags'].str.split().str.join(' ')
    df['tags'] = df['tags'].fillna('')

    # --- TF-IDF Calculation ---
    print("Calculating TF-IDF matrix...")
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=MAX_FEATURES)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['tags'])
    print("TF-IDF calculation complete.")

    # --- Cosine Similarity Calculation ---
    print("Calculating Cosine Similarity matrix...")
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    print("Cosine Similarity calculation complete.")

    # Reset index for mapping
    df_indexed = df.reset_index(drop=True) # drop=True prevents old index becoming a column
    indices_map = pd.Series(df_indexed.index, index=df_indexed['title'])

    print("Data loading and preparation finished.")
    return df_indexed, cosine_sim, indices_map

# --- Load data ONCE when the application starts ---
try:
    content_df, cosine_sim_matrix, indices = load_and_prepare_data(DATA_PATH)
    print(f"Successfully loaded and processed {len(content_df)} items.")
except FileNotFoundError as e:
    print(f"ERROR: {e}")
    print("Please ensure 'content_raw.csv' exists in the '../data' directory relative to 'webapp/app.py'.")
    # You might want to exit or have fallback logic if data loading fails critically
    content_df, cosine_sim_matrix, indices = pd.DataFrame(), None, pd.Series() # Empty defaults
except Exception as e:
    print(f"An unexpected error occurred during data loading: {e}")
    content_df, cosine_sim_matrix, indices = pd.DataFrame(), None, pd.Series()

# --- FastAPI App Instance ---
app = FastAPI()

# --- Mount Static Files Directory --- 
# This line tells FastAPI to serve files from the 'static' directory 
# when the URL starts with '/static'
# Make sure you have a 'static' directory in the same level as app.py
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Configure Templates ---
# Make sure you have a 'templates' directory in the same level as app.py
templates = Jinja2Templates(directory="templates")

# --- Helper Function to Get Poster URL ---
def get_poster_url(path, size="w300"):
    base_url = "https://image.tmdb.org/t/p/"
    if pd.isna(path) or path == '':
        # Return path to our static placeholder
        return "/static/placeholder.png" 
    return f"{base_url}{size}{path}"

# --- API Endpoints --- 

@app.get("/")
def read_root_html(request: Request): # Inject Request object
    """ Serves the main HTML page with popular items. """
    if content_df.empty:
        return templates.TemplateResponse("error.html", {"request": request, "message": "Data could not be loaded."}) # Need an error template

    # Get top N popular items (simple sort by popularity)
    popular_items = content_df.sort_values(by='popularity', ascending=False).head(24)
    # Add poster URLs to the data passed to the template
    popular_items_list = popular_items.to_dict("records")
    for item in popular_items_list:
        item['poster_url'] = get_poster_url(item.get('poster_path'))

    # Render the index.html template
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "popular_items": popular_items_list}
    )

@app.get("/popular")
def get_popular(limit: int = 20):
    """ API endpoint to get popular items (e.g., for dynamic loading) """
    if content_df.empty:
        raise HTTPException(status_code=503, detail="Content data not loaded.")
    
    popular = content_df.sort_values(by='popularity', ascending=False).head(limit)
    popular_list = popular.to_dict("records")
    for item in popular_list:
        item['poster_url'] = get_poster_url(item.get('poster_path'))
    return {"popular_items": popular_list}

# --- NEW Search Endpoint --- 
@app.get("/search")
def search_content(query: str):
    """
    Searches for content items by title (case-insensitive substring match).
    
    - **query**: The search term (query parameter).
    """
    if content_df.empty:
        raise HTTPException(status_code=503, detail="Content data not loaded.")
    
    if not query or len(query) < 2:
        # Avoid overly broad searches or empty queries
        return {"query": query, "results": []}

    print(f"Performing search for query: '{query}'") # Log search query
    try:
        # Perform case-insensitive substring search on the title
        # Using na=False to handle any potential unexpected NaNs in title column
        matches = content_df[content_df['title'].str.contains(query, case=False, na=False)]
        
        # Limit number of results
        max_results = 50 
        results_df = matches.head(max_results)
        
        results_list = results_df[['id', 'title', 'type', 'poster_path']].to_dict("records")
        
        # Add poster URLs
        for item in results_list:
            item['poster_url'] = get_poster_url(item.get('poster_path'))
            
        print(f"Found {len(results_list)} matches for query: '{query}'") # Log result count
        return {"query": query, "results": results_list}
        
    except Exception as e:
        print(f"Error during search for '{query}': {e}")
        raise HTTPException(status_code=500, detail="Internal server error during search.")

# --- NEW: Endpoint to get unique genres ---
@app.get("/genres")
def get_unique_genres():
    """ Returns a sorted list of unique genres from the dataset. """
    if content_df.empty:
        raise HTTPException(status_code=503, detail="Content data not loaded.")
    
    try:
        # Split comma-separated genres, handle potential NaNs/empties, flatten list
        all_genre_lists = content_df['genre_names'].dropna().str.split(',')
        # Flatten the list of lists and strip whitespace
        flat_genres = [genre.strip() for sublist in all_genre_lists for genre in sublist if genre.strip()]
        # Get unique, sorted genres
        unique_genres = sorted(list(set(flat_genres)))
        return {"genres": unique_genres}
    except Exception as e:
        print(f"Error getting unique genres: {e}")
        raise HTTPException(status_code=500, detail="Error processing genres.")

# --- NEW: Endpoint to get content by genre (with pagination) ---
@app.get("/genre/{genre_name}")
def get_content_by_genre(genre_name: str, skip: int = 0, limit: int = 20):
    """
    Gets content items belonging to a specific genre, with pagination.
    
    - **genre_name**: The genre to filter by (path parameter).
    - **skip**: Number of items to skip (for pagination).
    - **limit**: Maximum number of items to return.
    """
    if content_df.empty:
        raise HTTPException(status_code=503, detail="Content data not loaded.")
    
    if limit <= 0:
        limit = 20 # Default limit if invalid value provided
    if skip < 0:
        skip = 0
        
    try:
        # Filter by genre (case-insensitive contains match)
        # Adding word boundaries `\b` for more precise matching (e.g., prevent 'Action' matching 'Reaction')
        # Regex requires escaping special characters in genre_name if they exist, but unlikely for genres
        # Pandas `contains` needs regex=True for word boundaries
        pattern = rf'\b{genre_name}\b' #  = word boundary
        matches = content_df[content_df['genre_names'].str.contains(pattern, case=False, na=False, regex=True)]
        
        # Get total count for potential pagination UI later
        total_matches = len(matches)
        
        # Apply pagination
        paginated_matches = matches.iloc[skip : skip + limit]
        
        results_list = paginated_matches[['id', 'title', 'type', 'poster_path']].to_dict("records")
        
        # Add poster URLs
        for item in results_list:
            item['poster_url'] = get_poster_url(item.get('poster_path'))
            
        return {
            "genre": genre_name, 
            "skip": skip,
            "limit": limit,
            "total_matches": total_matches,
            "results": results_list
        }
        
    except Exception as e:
        print(f"Error getting content for genre '{genre_name}': {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error processing genre '{genre_name}'.")

# --- NEW Endpoint to get all content with pagination ---
@app.get("/all")
def get_all_content(skip: int = 0, limit: int = 24): # Default limit 24 for grid
    """
    Gets all content items with pagination, sorted by popularity.
    
    - **skip**: Number of items to skip.
    - **limit**: Maximum number of items to return.
    """
    if content_df.empty:
        raise HTTPException(status_code=503, detail="Content data not loaded.")
    
    if limit <= 0:
        limit = 24
    if skip < 0:
        skip = 0
        
    try:
        # Get subset sorted by popularity
        sorted_df = content_df.sort_values(by='popularity', ascending=False)
        total_items = len(sorted_df)
        
        # Apply pagination
        paginated_matches = sorted_df.iloc[skip : skip + limit]
        
        results_list = paginated_matches[['id', 'title', 'type', 'poster_path']].to_dict("records")
        
        # Add poster URLs
        for item in results_list:
            item['poster_url'] = get_poster_url(item.get('poster_path'))
            
        return {
            "skip": skip,
            "limit": limit,
            "total_items": total_items,
            "results": results_list
        }
        
    except Exception as e:
        print(f"Error getting all content: {e}")
        raise HTTPException(status_code=500, detail="Internal server error processing request.")

# --- NEW API Endpoint for Movies with Pagination ---
@app.get("/api/movies")
def get_movies(skip: int = 0, limit: int = 24):
    """
    Gets Movie items with pagination, sorted by popularity.
    """
    if content_df.empty:
        raise HTTPException(status_code=503, detail="Content data not loaded.")
    
    if limit <= 0:
        limit = 24
    if skip < 0:
        skip = 0
        
    try:
        # Filter for Movies first
        movies_df = content_df[content_df['type'].str.lower() == 'movie']
        
        # Sort by popularity
        sorted_movies = movies_df.sort_values(by='popularity', ascending=False)
        total_items = len(sorted_movies)
        
        # Apply pagination
        paginated_matches = sorted_movies.iloc[skip : skip + limit]
        
        results_list = paginated_matches[['id', 'title', 'type', 'poster_path']].to_dict("records")
        
        # Add poster URLs
        for item in results_list:
            item['poster_url'] = get_poster_url(item.get('poster_path'))
            
        return {
            "skip": skip,
            "limit": limit,
            "total_items": total_items,
            "results": results_list
        }
        
    except Exception as e:
        print(f"Error getting movies: {e}")
        raise HTTPException(status_code=500, detail="Internal server error processing movie request.")

# --- NEW API Endpoint for TV Shows with Pagination ---
@app.get("/api/shows")
def get_shows(skip: int = 0, limit: int = 24):
    """
    Gets TV Show items with pagination, sorted by popularity.
    NOTE: Assumes the type is stored as 'TV Show' in the data.
          Adjust the .str.lower() == 'tv show' if needed.
    """
    if content_df.empty:
        raise HTTPException(status_code=503, detail="Content data not loaded.")
    
    if limit <= 0:
        limit = 24
    if skip < 0:
        skip = 0
        
    try:
        # Filter for TV Shows first (using 'tv' as specified)
        shows_df = content_df[content_df['type'].str.lower() == 'tv'] 
        
        # Sort by popularity
        sorted_shows = shows_df.sort_values(by='popularity', ascending=False)
        total_items = len(sorted_shows)
        
        # Apply pagination
        paginated_matches = sorted_shows.iloc[skip : skip + limit]
        
        results_list = paginated_matches[['id', 'title', 'type', 'poster_path']].to_dict("records")
        
        # Add poster URLs
        for item in results_list:
            item['poster_url'] = get_poster_url(item.get('poster_path'))
            
        return {
            "skip": skip,
            "limit": limit,
            "total_items": total_items,
            "results": results_list
        }
        
    except Exception as e:
        print(f"Error getting TV shows: {e}")
        raise HTTPException(status_code=500, detail="Internal server error processing TV show request.")

# --- Recommendation Logic Function (adapted from notebook) ---
def get_recommendations_logic(title: str, top_n: int = 10):
    """Internal logic to get recommendations."""
    if content_df.empty or cosine_sim_matrix is None:
        raise HTTPException(status_code=503, detail="Recommendation data not loaded.")

    if title not in indices:
         # Basic check for close matches (case-insensitive substring)
        possible_matches = [t for t in indices.index if title.lower() in t.lower()]
        if not possible_matches:
            raise HTTPException(status_code=404, detail=f"Title '{title}' not found.")
        else:
            # In a real app, might return suggestions. Here, use the first match.
            title = possible_matches[0]

    idx = indices[title]
    if isinstance(idx, pd.Series): # Handle potential duplicate titles mapping to multiple indices
        idx = idx.iloc[0]

    sim_scores = list(enumerate(cosine_sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in sim_scores[1:top_n + 1]]

    # Return titles and maybe type/id for more usefulness
    results = content_df.iloc[top_indices][['id', 'title', 'type', 'poster_path']].to_dict("records") # Added poster_path
    # Add poster URLs to recommendations
    for item in results:
        item['poster_url'] = get_poster_url(item.get('poster_path'))
    return results

# --- Recommendation API Endpoint --- 
@app.get("/recommend/{title}")
def recommend(title: str, top_n: int = 10):
    """
    Provides top N content recommendations for a given title.
    
    - **title**: The movie or TV show title (path parameter).
    - **top_n**: The number of recommendations to return (query parameter, default 10).
    """
    try:
        recommendations = get_recommendations_logic(title, top_n)
        return {"input_title": title, "recommendations": recommendations}
    except HTTPException as e:
        # Re-raise HTTPException to let FastAPI handle it
        raise e
    except Exception as e:
        # Catch unexpected errors during recommendation
        print(f"Error during recommendation for '{title}': {e}") # Log the error server-side
        raise HTTPException(status_code=500, detail="Internal server error during recommendation.")

# --- Item Details Endpoint ---
@app.get("/item/{item_id}")
def get_item_details(item_id: str):
    """
    Get detailed information for a specific content item by ID.
    
    - **item_id**: The ID of the movie or TV show to retrieve.
    """
    if content_df.empty:
        raise HTTPException(status_code=503, detail="Content data not loaded.")
    
    try:
        # Convert string ID to proper type for comparison
        if item_id.isdigit():
            # Try both string and int comparisons for flexibility
            item = content_df[(content_df['id'] == int(item_id)) | (content_df['id'] == item_id)]
        else:
            # For non-numeric IDs
            item = content_df[content_df['id'] == item_id]
        
        if item.empty:
            raise HTTPException(status_code=404, detail=f"Item with ID {item_id} not found.")
        
        # Convert to dict and add poster URL
        item_dict = item.iloc[0].to_dict()
        item_dict['poster_url'] = get_poster_url(item_dict.get('poster_path'))
        
        return item_dict
        
    except HTTPException as e:
        # Re-raise HTTPException to let FastAPI handle it
        raise e
    except Exception as e:
        print(f"Error retrieving item details for ID {item_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error retrieving item with ID {item_id}.")

# To run: uvicorn app:app --reload

# --- Placeholder for future routes ---
# @app.get("/items/{item_id}")
# async def read_item(item_id: int, q: str | None = None):
#     return {"item_id": item_id, "q": q}

# To run this application:
# 1. Navigate to the 'webapp' directory in your terminal.
# 2. Run the command: uvicorn app:app --reload
#    - 'app:app' refers to the file app.py and the 'app' instance inside it.
#    - '--reload' makes the server restart automatically when you save changes to the code.
# 3. Open your web browser and go to http://127.0.0.1:8000 