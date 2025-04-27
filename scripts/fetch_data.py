#! /usr/bin/env python3

import requests
import pandas as pd
import os
import time
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

if not TMDB_API_KEY:
    raise ValueError("TMDB_API_KEY not found. Please set it in a .env file.")

BASE_URL = "https://api.themoviedb.org/3"
FETCH_DELAY = 0.5 # Seconds delay between API calls

def get_genre_map(content_type='movie'):
    """Fetches the genre ID to name mapping for movies or TV from TMDB."""
    if content_type not in ['movie', 'tv']:
        raise ValueError("content_type must be 'movie' or 'tv'")

    url = f"{BASE_URL}/genre/{content_type}/list?api_key={TMDB_API_KEY}&language=en-US"
    print(f"Fetching {content_type} genre map...")
    try:
        response = requests.get(url)
        response.raise_for_status() # Raises HTTPError for bad responses (4XX or 5XX)
        genres = response.json().get('genres', [])
        print(f"Successfully fetched {content_type} genre map.")
        return {genre['id']: genre['name'] for genre in genres}
    except requests.exceptions.RequestException as e:
        print(f"!!! Error fetching {content_type} genres: {e}")
        return {}

def fetch_popular_content(content_type, num_pages, genre_map):
    """Fetches popular content (movies or TV) from TMDB API."""
    all_content = []
    content_path = 'movie' if content_type == 'movie' else 'tv' # API path segment

    print(f"\nAttempting to fetch {num_pages} pages of popular {content_type} shows...")
    for page in range(1, num_pages + 1):
        url = f"{BASE_URL}/{content_path}/popular?api_key={TMDB_API_KEY}&language=en-US&page={page}"
        try:
            response = requests.get(url)
            if response.status_code == 429:
                 print(f"Rate limit hit on page {page}. Waiting longer...")
                 time.sleep(5)
                 response = requests.get(url)

            response.raise_for_status()
            content_data = response.json()
            results = content_data.get('results', [])
            if not results:
                 print(f"No results found on page {page} for {content_type}. Reached end.")
                 break

            for item in results:
                genre_names = [genre_map.get(gid) for gid in item.get('genre_ids', []) if genre_map.get(gid)]
                
                # Standardize fields
                item_id = item.get('id')
                title = item.get('title') if content_type == 'movie' else item.get('name')
                release_date = item.get('release_date') if content_type == 'movie' else item.get('first_air_date')
                overview = item.get('overview')
                vote_average = item.get('vote_average')
                vote_count = item.get('vote_count')
                popularity = item.get('popularity')
                poster_path = item.get('poster_path')

                all_content.append({
                    'id': item_id,
                    'title': title,
                    'overview': overview,
                    'release_date': release_date,
                    'vote_average': vote_average,
                    'vote_count': vote_count,
                    'popularity': popularity,
                    'poster_path': poster_path,
                    'genre_names': ', '.join(genre_names),
                    'type': content_type # Add type column
                })
            print(f"Successfully fetched page {page}/{num_pages} for {content_type}. Total {content_type} collected: {len(all_content)}")
            time.sleep(FETCH_DELAY)

        except requests.exceptions.RequestException as e:
            print(f"\n!!! Error fetching {content_type} page {page}: {e}")
            print(f"    URL: {url}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"    Status Code: {e.response.status_code}")
                print(f"    Response Text: {e.response.text[:200]}...")
            print(f"    Stopping fetch process for {content_type} due to error.\n")
            break

    return pd.DataFrame(all_content)

if __name__ == "__main__":
    # --- Configuration ---
    NUM_PAGES_MOVIES = 100 # Number of pages for movies (20 items/page)
    NUM_PAGES_TV = 100     # Number of pages for TV shows

    # --- Setup ---
    data_dir = '../data'
    if not os.path.exists(data_dir):
        print(f"Creating directory: {data_dir}")
        try:
            os.makedirs(data_dir)
            print("Directory created successfully.")
        except OSError as e:
            print(f"!!! ERROR creating directory {data_dir}: {e} !!!")
            exit(1) # Exit if directory creation fails

    # --- Fetch Genre Maps ---
    movie_genre_map = get_genre_map('movie')
    tv_genre_map = get_genre_map('tv')

    all_dataframes = []

    # --- Fetch Movies ---
    if movie_genre_map:
        movies_df = fetch_popular_content('movie', NUM_PAGES_MOVIES, movie_genre_map)
        if not movies_df.empty:
            print(f"\nFinished fetching movies. Collected {len(movies_df)} entries.")
            all_dataframes.append(movies_df)
        else:
            print("\nNo movies were fetched.")
    else:
        print("\nCould not fetch movie genre map. Skipping movie fetch.")

    # --- Fetch TV Shows ---
    if tv_genre_map:
        tv_df = fetch_popular_content('tv', NUM_PAGES_TV, tv_genre_map)
        if not tv_df.empty:
            print(f"\nFinished fetching TV shows. Collected {len(tv_df)} entries.")
            all_dataframes.append(tv_df)
        else:
            print("\nNo TV shows were fetched.")
    else:
        print("\nCould not fetch TV genre map. Skipping TV fetch.")

    # --- Combine and Save ---
    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        print(f"\nCombined movies and TV shows. Total entries: {len(combined_df)}")

        # Optional: Drop duplicates that might arise if an item is somehow fetched twice
        combined_df = combined_df.drop_duplicates(subset=['id', 'type'])
        print(f"Shape after dropping potential duplicates: {combined_df.shape}")

        output_path = os.path.join(data_dir, 'content_raw.csv') # New filename
        print(f"\nAttempting to save combined data to: {output_path}")
        try:
            combined_df.to_csv(output_path, index=False)
            print(f"Successfully saved combined data to {output_path}")
            print("\nFinal combined data - First 5 rows:")
            print(combined_df.head())
        except Exception as e:
            print(f"\n!!! ERROR occurred during .to_csv(): {e} !!!\n")
    else:
        print("\nNo data was fetched from either movies or TV shows. No file saved.")

    print("\nScript finished.") 