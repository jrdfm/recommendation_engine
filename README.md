# Netflix-Style Content Recommender

A movie and TV show recommendation system with a Netflix-inspired UI.

## Features

- Browse all content with infinite scrolling
- Netflix-style sidebar navigation
- Content recommendation based on similarity
- Search functionality
- Filter by genre

## Prerequisites

- Python 3.7+
- FastAPI
- pandas
- scikit-learn

## Setup

1. Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the required packages:

```bash
pip install fastapi uvicorn pandas scikit-learn jinja2
```

3. Make sure you have the content data file in the correct location:
   - The application expects a file called `content_raw.csv` in the `data` directory

## Running the Application

Navigate to the `webapp` directory and run:

```bash
cd webapp
uvicorn app:app --reload
```

The application will be available at [http://localhost:8000](http://localhost:8000)

## Project Structure

- `webapp/`: Main application directory
  - `app.py`: FastAPI application code
  - `templates/`: HTML templates
  - `static/`: Static files (CSS, images)
- `data/`: Data files
- `scripts/`: Utility scripts

## Screenshots

The application mimics the Netflix interface with a dark theme and sidebar navigation. 