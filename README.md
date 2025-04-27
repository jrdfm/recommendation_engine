# Content Recommender

A web application demonstrating a content-based recommendation system for movies and TV shows, built with FastAPI and featuring a dynamic user interface.

## Features

- **Dynamic UI:** Single-page application feel using JavaScript to load content without full page reloads.
- **Content Browsing:** View all Movies or TV Shows with infinite scrolling.
- **Genre Filtering:** Browse content filtered by specific genres.
- **Content-Based Recommendations:** Get recommendations similar to a selected title or based on items in "My List".
- **My List:** Add/remove items to a persistent "My List" stored in browser `localStorage`.
- **Item Details:** View detailed information about a specific movie or show in a modal view.
- **Search:** Search for content by title keywords.

## Technical Implementation

### Backend (FastAPI - `webapp/app.py`)

The backend is built using the FastAPI framework.

- **Data Loading & Preprocessing (`load_and_prepare_data`):**
    - Reads content data from a CSV file specified by `DATA_PATH` (currently `../data/content_raw.csv`).
    - Performs basic cleaning: drops duplicates, removes rows with missing essential data (title, overview, genre).
    - Creates a `tags` feature by combining lowercased title, overview, and genre names for TF-IDF.
    - **TF-IDF Vectorization:** Uses `sklearn.feature_extraction.text.TfidfVectorizer` to convert the `tags` into a numerical matrix, considering stop words and limiting features (`MAX_FEATURES`).
    - **Cosine Similarity:** Calculates the pairwise cosine similarity between all items using the TF-IDF matrix (`sklearn.metrics.pairwise.cosine_similarity`). This matrix forms the basis of the content-based recommendations.
    - An `indices` mapping (Pandas Series) is created to map content titles to their index in the DataFrame/similarity matrix for quick lookups.
    - This entire process runs once at application startup.

- **API Endpoints:**
    - `/` (GET): Serves the main `index.html` template.
    - `/popular` (GET): Returns a list of the most popular items (sorted by `popularity` column), used for the home page.
    - `/api/movies` (GET): Returns a paginated list of all items with `type` == 'movie', sorted by popularity.
    - `/api/shows` (GET): Returns a paginated list of all items with `type` == 'tv', sorted by popularity.
    - `/genres` (GET): Returns a sorted list of unique genre names extracted from the dataset.
    - `/genre/{genre_name}` (GET): Returns a paginated list of items matching the specified `genre_name`.
    - `/search` (GET): Performs a case-insensitive substring search on titles based on the `query` parameter.
    - `/recommend/{title}` (GET): Provides content recommendations based on the provided `title`.
    - `/item/{item_id}` (GET): Retrieves detailed information for a specific item by its ID.
    - **Pagination:** Endpoints returning lists (`/api/movies`, `/api/shows`, `/genre/...`) support `skip` and `limit` query parameters for pagination.

- **Recommendation Logic (`get_recommendations_logic`):**
    - Takes a `title` and `top_n` number of recommendations to return.
    - Uses the `indices` map to find the index of the input `title`.
    - Retrieves the corresponding row from the precomputed `cosine_sim_matrix`.
    - Sorts items based on their similarity score to the input title.
    - Returns the `top_n` most similar items (excluding the input item itself).

- **Error Handling:** Uses FastAPI's `HTTPException` for standard HTTP error responses (e.g., 404 Not Found, 503 Service Unavailable if data isn't loaded).

### Frontend (HTML/CSS/JavaScript - `webapp/templates/index.html`)

The frontend is a single HTML file using Jinja2 templating (though minimally), embedded CSS, and extensive vanilla JavaScript for interactivity.

- **Structure:** Uses a flexbox layout (`.page-wrapper`) with a fixed sidebar (`#sidebar`) and a main content area (`#main-content`). The `#content-display-area` within `#main-content` is dynamically updated by JavaScript.
- **Dynamic Content Loading:** Clicks on sidebar navigation items (`#nav-home`, `#nav-shows`, `#nav-movies`, `#nav-mylist`, genre links) trigger JavaScript functions.
    - These functions typically clear the `#content-display-area` and call other functions (`loadHomePage`, `loadAllContent`, `loadMyList`, `fetchAndDisplayGenre`) to fetch data from the corresponding backend API endpoints using the `fetch` API.
    - Results are rendered into grid layouts (`.grid`) using the `displayItems` function.
- **Home Page (`loadHomePage`):**
    - Asynchronously loads multiple sections:
        - **My List:** Shows the first 10 items from local storage. Includes a "View All" link if > 10 items.
        - **Recommendations:** Selects up to 5 items from "My List" (randomly if > 5). For each selected item, it fetches 10 recommendations using `/recommend/{title}` and displays them in a dedicated "Because you liked..." section.
        - **Popular:** Fetches and displays the top 10 popular items from `/popular`.
    - Uses `createContentSection` to build the HTML structure for each section, initially showing a "Loading..." message.
    - Uses `displayItems` to populate the sections once data is fetched, removing the loading message.
- **Content Display (`displayItems`):** Renders fetched items into a grid, creating elements for poster images, titles, and type icons. Adds click listeners to grid items to trigger the details modal.
- **Infinite Scroll (`handleScroll`):**
    - Listens for scroll events.
    - Uses debouncing (`setTimeout`) to limit checks.
    - When scrolling near the bottom of the page *and* the current view supports it (`movies`, `shows`, `genre`), it calls `fetchAndDisplayBatch` or `fetchAndDisplayGenre` with an incremented `skip` value to load the next batch of items.
- **My List (`getMyList`, `saveMyList`, `toggleMyListItem`, `isItemInMyList`):** Uses browser `localStorage` to persist the user's list of saved items between sessions.
- **Modal (`#details-modal`, `openModal`, `closeModal`, `showItemDetails`):** Handles displaying detailed item information fetched from the `/item/{item_id}` endpoint in a pop-up modal.
- **State Management:** Uses simple JavaScript variables (`currentView`, `currentGenre`, `currentSkip`, `isLoading`, etc.) to manage the application's state.

## Prerequisites

- Python 3.7+
- pip (Python package installer)

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Linux/macOS:
    source venv/bin/activate
    # On Windows:
    # venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install fastapi uvicorn pandas scikit-learn jinja2
    ```
    *(Consider adding these to a `requirements.txt` file)*

4.  **Data File:**
    - Ensure the data file `content_raw.csv` exists in the `data/` directory at the project root (`rec/data/content_raw.csv`).
    - This file should contain columns like `id`, `title`, `overview`, `type` ('movie' or 'tv'), `genre_names`, `popularity`, `poster_path`.

## Running the Application

1.  Navigate to the `webapp` directory:
    ```bash
    cd webapp
    ```

2.  Start the FastAPI server using Uvicorn:
    ```bash
    uvicorn app:app --reload
    ```
    - The `--reload` flag automatically restarts the server when code changes are detected.

3.  Open your web browser and go to [http://127.0.0.1:8000](http://127.0.0.1:8000)

## Project Structure

```
rec/
├── data/
│   └── content_raw.csv     # Main data file (or your data file)
├── webapp/
│   ├── static/
│   │   └── ... (CSS, images, e.g., netflix-logo.svg, placeholder.png)
│   ├── templates/
│   │   └── index.html      # Main HTML template with CSS and JS
│   └── app.py            # FastAPI backend code
├── venv/                   # Virtual environment (if created)
└── README.md             # This file
```

## Screenshots
