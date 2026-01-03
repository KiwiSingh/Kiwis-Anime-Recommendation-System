# 🥝 Kiwi's Anime Recommendation System

A sophisticated anime recommendation system built with Streamlit that leverages multiple anime databases (MyAnimeList, Kitsu, AniList), natural language processing, and machine learning to provide personalized anime recommendations.

## Features

- **Multi-API Integration**: Combines data from MyAnimeList (MAL), Kitsu, and AniList for comprehensive anime information
- **Natural Language Processing**: Uses sentiment analysis and TF-IDF vectorization to understand user preferences
- **Special Collections**: Pre-defined collections for iconic anime themes (Pokemon, Digimon, Vampire, etc.)
- **Interactive UI**: Built with Streamlit for an intuitive user experience
- **OAuth Authentication**: Secure login with MyAnimeList for personalized recommendations
- **Visual Analytics**: Interactive charts and graphs using Plotly
- **Smart Filtering**: Advanced filtering based on genres, ratings, and user preferences

## Installation

1. Clone this repository:
   ```bash
   git clone https://www.github.com/KiwiSingh/Kiwis-Anime-Recommendation-System
   cd Kiwis-Anime-Recommendation-System
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

### MyAnimeList API Setup
To enable MyAnimeList authentication and personalized features:

1. Go to [MyAnimeList Developer Portal](https://myanimelist.net/apiconfig)
2. Create a new application
3. Set the redirect URI to: `http://localhost:8501/?popup=true`
4. Copy your Client ID and Client Secret

### Environment Variables
Create a `.streamlit/secrets.toml` file in your project directory:

```toml
MAL_CLIENT_ID = "your_client_id_here"
MAL_CLIENT_SECRET = "your_client_secret_here"
```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open your browser to `http://localhost:8501`

3. Choose your preferred anime provider (MyAnimeList, Kitsu, or AniList)

4. Describe your anime preferences in natural language or select from special collections

5. Get personalized recommendations with detailed information and analytics

## Requirements

- Python 3.8+
- Internet connection for API calls
- MyAnimeList account (optional, for personalized features)

## Dependencies

- streamlit: Web app framework
- jikanpy: MyAnimeList API wrapper
- requests-oauthlib: OAuth2 authentication
- vaderSentiment: Sentiment analysis
- scikit-learn: Machine learning algorithms
- pandas: Data manipulation
- plotly: Interactive visualizations
- requests: HTTP requests

## Architecture

The system uses:
- **Content-based filtering** with TF-IDF and cosine similarity
- **Sentiment analysis** to gauge user preferences
- **Multi-source data aggregation** for robust recommendations
- **Rate limiting handling** with exponential backoff
- **Session state management** for user experience

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is open source. Please check the license file for details.
