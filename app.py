import streamlit as st  # type: ignore
from typing import List, Dict, Any, Optional, Tuple
from requests_oauthlib import OAuth2Session  # type: ignore
from jikanpy import Jikan  # type: ignore
from jikanpy.exceptions import APIException  # type: ignore
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
import requests  # type: ignore
import pandas as pd  # type: ignore
import plotly.express as px  # type: ignore
import time
import secrets
import re
from collections import Counter
import logging
import streamlit.components.v1 as components # type: ignore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Kiwi's Anime Recommendation System",
    page_icon="🥝",
    layout="wide"
)

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'current_list': [],
        'manual_favs': [],
        'provider_display_name': "",
        'nlp_vibe': "",
        'mal_authenticated': False,
        'selected_provider': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# API Constants
MAL_CLIENT_ID = st.secrets.get("MAL_CLIENT_ID", "")
MAL_CLIENT_SECRET = st.secrets.get("MAL_CLIENT_SECRET", "")
REDIRECT_URI = "https://kiwi-anime-recs.streamlit.app/"
KITSU_REST_URL = "https://kitsu.io/api/edge"
ANILIST_URL = "https://graphql.anilist.co"

# Default headers for API requests
DEFAULT_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

# Handle OAuth callback from popup
if 'code' in st.query_params and 'state' in st.query_params:
    code = st.query_params['code']
    state = st.query_params.get('state', '')
    
    # If opened in a popup, redirect parent and close
    components.html(f"""
    <script>
    if (window.opener) {{
        // Redirect parent window to the code callback URL
        const parentUrl = window.location.origin + window.location.pathname + '?code={code}&state={state}';
        window.opener.location.href = parentUrl;
        // Close this popup window
        window.close();
    }}
    </script>
    """)
    st.stop()

# --- SPECIAL ANIME COLLECTIONS ---
ICONIC_COLLECTIONS = {
    'pokemon': {
        'search_terms': ['pokemon', 'pocket monsters', 'pikachu'],
        'mal_ids': [527, 19291, 9107, 20159, 37117, 40351, 55701],
        'keywords': ['pokemon', 'pocket monster', 'pikachu', 'ash ketchum', 'satoshi']
    },
    'digimon': {
        'search_terms': ['digimon', 'digital monsters'],
        'mal_ids': [552, 1316, 1334, 1368, 3888, 5152],
        'keywords': ['digimon', 'digital monster', 'agumon', 'tai']
    },
    'vampire': {
        'search_terms': ['vampire', 'hellsing', 'blood', 'dracula'],
        'mal_ids': [28, 777, 355, 6920, 3457, 11111],
        'keywords': ['vampire', 'blood', 'dracula', 'immortal']
    },
    'cgdct': {
        'search_terms': ['k-on', 'yuru camp', 'non non biyori', 'school', 'girls', 'comedy', 'slice of life'],
        'mal_ids': [17549, 5680, 34798, 23587, 28999, 30831, 38656, 20047, 10165, 15051],
        'keywords': ['cute girls doing cute things', 'cgdct', 'moe', 'iyashikei', 'girls doing cute']
    },
    'yuri': {
        'search_terms': ['yuri', 'girls love', 'shoujo ai', 'bloom into you', 'citrus', 'lesbian'],
        'mal_ids': [37786, 34382, 50739, 39790, 20047, 6164, 32681, 21939, 44774],
        'keywords': ['yuri', 'girls love', 'shoujo ai', 'lesbian', 'girls doing cute girls', 'girl love']
    },
    'dogs': {
        'search_terms': ['dog', 'dogs', 'puppy', 'inukai', 'chainsaw man'],
        'mal_ids': [38000, 50709, 5114, 11757, 44511, 56965],
        'keywords': ['dog', 'puppy', 'canine', 'good boy', 'good boi']
    },
    'music': {
        'search_terms': ['music', 'band', 'k-on', 'bocchi', 'idol', 'song'],
        'mal_ids': [5680, 54112, 457, 377, 15051, 22789],
        'keywords': ['music', 'band', 'idol', 'song', 'concert', 'instrument']
    },
    'tank': {
        'search_terms': ['panzer', 'girls und panzer', 'tank', 'military girls'],
        'mal_ids': [14131, 18617, 25391],
        'keywords': ['tank', 'tanks', 'panzer', 'sensha', 'military vehicle']
    }
}

# --- UTILITY FUNCTIONS ---
def safe_jikan_call(func, *args, max_retries: int = 3, **kwargs):
    """Handles Jikan API rate limits with exponential backoff"""
    base_delay = 0.5
    
    for attempt in range(max_retries):
        try:
            if attempt == 0:
                time.sleep(0.3)
            else:
                time.sleep(base_delay * (2 ** attempt))
            
            return func(*args, **kwargs)
            
        except APIException as e:
            error_str = str(e)
            if "429" in error_str:
                wait_time = base_delay * (2 ** (attempt + 2))
                logger.warning(f"Rate limited, waiting {wait_time}s...")
                if attempt < max_retries - 1:
                    st.warning(f"⚠️ Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    return None
            elif "404" in error_str:
                return None
            else:
                logger.error(f"API Error: {error_str}")
                return None
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return None
    
    return None

def detect_special_collection(vibe: str) -> Optional[str]:
    """Detect if the vibe matches a special anime collection"""
    vibe_lower = vibe.lower()
    
    for collection_name, collection_data in ICONIC_COLLECTIONS.items():
        keywords = collection_data['keywords']
        if any(keyword in vibe_lower for keyword in keywords):
            return collection_name
    
    return None

def fetch_iconic_anime(collection_name: str, progress_callback=None) -> List[Dict]:
    """Fetch anime from iconic collections by MAL ID"""
    if collection_name not in ICONIC_COLLECTIONS:
        return []
    
    collection = ICONIC_COLLECTIONS[collection_name]
    results = []
    
    if progress_callback:
        progress_callback(f"📥 Fetching iconic {collection_name} anime...")
    
    jikan = Jikan()
    for mal_id in collection['mal_ids']:
        resp = safe_jikan_call(jikan.anime, mal_id)
        if resp and resp.get('data'):
            results.append(resp['data'])
            time.sleep(0.3)
    
    return results

# --- NLP PROCESSING ---
jikan = Jikan()
vader = SentimentIntensityAnalyzer()

def extract_key_concepts(user_query: str) -> Tuple[List[str], List[List[str]]]:
    """Extract must-have concepts and rare/specific terms from query"""
    query_lower = user_query.lower()
    
    stop_words = {
        'cute', 'with', 'and', 'anime', 'the', 'a', 'an', 'of', 'in', 'on',
        'for', 'to', 'is', 'are', 'about', 'like', 'show', 'series'
    }
    
    all_words = re.findall(r'\b\w+\b', query_lower)
    rare_terms = [w for w in all_words if w not in stop_words and len(w) > 2]
    
    key_concepts = []
    
    # Military/vehicle concepts
    military_terms = ['tank', 'tanks', 'military', 'war', 'vehicle', 'mecha', 'robot']
    if any(term in query_lower for term in military_terms):
        if 'tank' in query_lower or 'tanks' in query_lower:
            key_concepts.append(['tank', 'tanks', 'sensha', 'panzer'])
        else:
            key_concepts.append(military_terms)
    
    # Character type concepts
    if 'girl' in query_lower or 'girls' in query_lower:
        key_concepts.append(['girl', 'girls', 'female', 'schoolgirl'])
    if 'boy' in query_lower or 'boys' in query_lower:
        key_concepts.append(['boy', 'boys', 'male'])
    
    # Monster concepts
    monster_terms = ['pokemon', 'digimon', 'monster', 'monsters', 'creature', 'creatures']
    if any(term in query_lower for term in monster_terms):
        key_concepts.append(monster_terms)
    
    # Music concepts
    music_terms = ['band', 'music', 'musical', 'song', 'instrument', 'concert']
    if any(term in query_lower for term in music_terms):
        key_concepts.append(music_terms)
    
    # Sports concepts
    sports_terms = ['sport', 'sports', 'basketball', 'volleyball', 'soccer', 'baseball', 'tennis']
    if any(term in query_lower for term in sports_terms):
        key_concepts.append(sports_terms)
    
    return rare_terms, key_concepts

def extract_base_title(title: str) -> str:
    """Extract the base title without sequel indicators"""
    title_lower = title.lower()
    
    # Remove common sequel indicators
    sequel_markers = [
        ' 2nd season', ' 3rd season', ' 4th season', ' 5th season',
        ' season 2', ' season 3', ' season 4', ' season 5',
        ' season ii', ' season iii', ' season iv', ' season v',
        ': 2nd season', ': 3rd season', ': season 2', ': season 3',
        '!!', '!!!', '!!!!',
        ' part 2', ' part 3', ' part 4', ' part 5',
        ' part ii', ' part iii', ' part iv', ' part v',
        ' ii', ' iii', ' iv', ' v',
        ' kan', ' zoku', 'repeat', 'reload', 'revolution', 'reloaded', 'nonstop',
        '♪♪', '♪♪♪', '△'
    ]
    
    base = title_lower
    for marker in sequel_markers:
        base = base.replace(marker, '')
    
    # Remove trailing numbers
    base = re.sub(r'\s+\d+$', '', base)
    
    # Remove trailing symbols
    base = base.rstrip('!♪△▲')
    
    return base.strip()

def compute_nlp_similarity(user_query: str, anime_list: List[Dict]) -> List[Dict]:
    """Uses TF-IDF Vectorization with improved keyword boosting"""
    if not user_query or not anime_list:
        return anime_list

    rare_terms, key_concepts = extract_key_concepts(user_query)
    requires_all_concepts = len(key_concepts) >= 2

    # Extract negative keywords
    negative_keywords = []
    query_lower = user_query.lower()
    negation_patterns = [
        'not ', 'no ', 'without ', 'avoid ', "don't ", 'except ', 'but not ', 'minus '
    ]
    
    for pattern in negation_patterns:
        if pattern in query_lower:
            parts = query_lower.split(pattern)
            if len(parts) > 1:
                next_words = parts[1].split()[:3]
                negative_keywords.extend([w.strip('.,!?') for w in next_words])
    
    # Genre aliases
    genre_aliases = {
        'harem': ['harem', 'reverse harem'],
        'ecchi': ['ecchi', 'erotica'],
        'romance': ['romance', 'romantic'],
        'fanservice': ['ecchi', 'fan service']
    }
    
    expanded_negatives = set(negative_keywords)
    for keyword in negative_keywords:
        if keyword in genre_aliases:
            expanded_negatives.update(genre_aliases[keyword])
    
    needs_harem_detection = 'harem' in expanded_negatives
    
    # Detect search types
    is_yuri_search = any(term in query_lower for term in ['yuri', 'girls love', 'shoujo ai', 'lesbian', 'girls doing cute girls', 'girl love'])
    is_music_search = any(term in query_lower for term in ['band', 'music', 'idol', 'song', 'concert', 'instrument']) and 'cute girls doing cute' not in query_lower
    is_cgdct_search = any(term in query_lower for term in ['cute girls doing cute things', 'cgdct', 'girls doing cute'])
    
    # Sequel detection patterns
    sequel_indicators = [
        ' 2nd season', ' 3rd season', ' 4th season', ' 5th season',
        ' season 2', ' season 3', ' season 4', ' season 5',
        ' season ii', ' season iii', ' season iv', ' season v',
        ': 2nd season', ': 3rd season', ': season 2', ': season 3',
        '!!', '!!!', '!!!!',
        ' part 2', ' part 3', ' part 4', ' part 5',
        ' part ii', ' part iii', ' part iv', ' part v',
        'recap', 'picture drama', ': specials',
        ' kan', ' zoku', '속편',
        ' ii ', ' iii ', ' iv ', ' v ',
        'repeat', 'reload', 'revolution', 'reloaded', 'nonstop',
        '♪♪', '♪♪♪',
        '2nd', '3rd', '4th', '5th'
    ]
    
    # Harem detection patterns
    harem_indicators = [
        'harem', 'girls', 'female', 'women', 'ladies',
        'love triangle', 'romantic', 'romance', 'feelings for',
        'attracted to', 'affection', 'falls for', 'crush on',
        'pretty', 'beautiful', 'cute girls', 'attractive',
        'admirer', 'suitor', 'love interest',
        'competing', 'jealous', 'rivalry'
    ]
    
    strong_harem_signals = [
        'harem', 'love triangle', 'girls compete', 'reverse harem',
        'multiple love interests', 'romantic advances', 'polygamy',
        'attracted to him', 'vying for', 'fighting for his'
    ]
    
    known_harem_franchises = [
        're:zero', 'rezero', 'mushoku tensei', 'sword art online', 'sao',
        'date a live', 'high school dxd', 'to love', 'nisekoi',
        'infinite stratos', 'trinity seven', 'campione', 'strike the blood',
        'testament', 'smartphone', 'maken-ki', 'shinmai maou',
        'grisaia', 'shuffle', 'amagami', 'quintessential quintuplets',
        'rent-a-girlfriend', 'girlfriend girlfriend', 'hundred',
        'asterisk war', 'chivalry of a failed', 'world break'
    ]
    
    documents = []
    valid_anime = []
    filtered_count = 0
    filter_reasons = []
    seen_base_titles = {}
    
    for a in anime_list:
        genres = [g.get('name', '').lower() for g in (a.get('genres') or [])]
        genre_str = " ".join(genres)
        title = str(a.get('title') or a.get('title_english') or '').lower()
        title_original = str(a.get('title') or a.get('title_english') or '')
        anime_type = str(a.get('type') or '').lower()
        
        # Filter non-TV content
        if anime_type in ['movie', 'ova', 'special', 'ona']:
            filtered_count += 1
            filter_reasons.append(f"{title[:30]}: {anime_type}")
            continue
        
        # STRICT YURI FILTERING
        if is_yuri_search:
            has_yuri_genre = 'girls love' in genres or 'shoujo ai' in genres
            if not has_yuri_genre:
                filtered_count += 1
                filter_reasons.append(f"{title[:30]}: not girls love genre")
                continue
            if 'romance' in genres and 'harem' in genres:
                filtered_count += 1
                filter_reasons.append(f"{title[:30]}: has harem tag")
                continue
        
        # STRICT MUSIC FILTERING
        if is_music_search:
            has_music_genre = 'music' in genres
            if not has_music_genre:
                filtered_count += 1
                filter_reasons.append(f"{title[:30]}: not music genre")
                continue
            if 'sports' in genres:
                filtered_count += 1
                filter_reasons.append(f"{title[:30]}: sports anime")
                continue
        
        # CGDCT FILTERING
        if is_cgdct_search:
            if 'music' in genres:
                filtered_count += 1
                filter_reasons.append(f"{title[:30]}: music anime")
                continue
            if 'girls love' in genres or 'shoujo ai' in genres:
                filtered_count += 1
                filter_reasons.append(f"{title[:30]}: yuri anime")
                continue
            if 'idol' in title or 'idolm@ster' in title or 'love live' in title:
                filtered_count += 1
                filter_reasons.append(f"{title[:30]}: idol anime")
                continue
        
        # Check for sequels
        is_sequel = False
        title_padded = f" {title} "
        
        for indicator in sequel_indicators:
            if indicator in title:
                if indicator in ['!!', '!!!', '!!!!']:
                    if title.count('!') >= 2:
                        is_sequel = True
                        break
                elif indicator in ['♪♪', '♪♪♪']:
                    if indicator in title_original:
                        is_sequel = True
                        break
                elif indicator in [' ii ', ' iii ', ' iv ', ' v ']:
                    if indicator in title_padded:
                        is_sequel = True
                        break
                else:
                    is_sequel = True
                    break
        
        ending_number = re.search(r'\s+(\d+)$', title)
        if ending_number and int(ending_number.group(1)) > 1:
            is_sequel = True
        
        if is_sequel:
            filtered_count += 1
            filter_reasons.append(f"{title[:30]}: sequel/season")
            continue
        
        # Franchise deduplication
        base_title = extract_base_title(title_original)
        if base_title in seen_base_titles:
            existing_anime = seen_base_titles[base_title]
            existing_score = existing_anime.get('score') or 0
            current_score = a.get('score') or 0
            
            if current_score > existing_score:
                valid_anime.remove(existing_anime)
                documents.remove(seen_base_titles[base_title + '_doc'])
                seen_base_titles[base_title] = a
            else:
                filtered_count += 1
                filter_reasons.append(f"{title[:30]}: duplicate franchise")
                continue
        else:
            seen_base_titles[base_title] = a
        
        synopsis = str(a.get('synopsis') or '').lower()
        themes = [t.get('name', '').lower() for t in (a.get('themes') or [])]
        themes_str = " ".join(themes)
        
        # Genre matching for negative filters
        genre_match = any(neg.lower() in genres for neg in expanded_negatives if neg)
        
        # Harem detection
        is_likely_harem = False
        harem_debug = ""
        if needs_harem_detection:
            title_check = any(franchise in title for franchise in known_harem_franchises)
            strong_match = any(signal in synopsis for signal in strong_harem_signals)
            found_indicators = [indicator for indicator in harem_indicators if indicator in synopsis]
            harem_score = len(found_indicators)
            female_count = synopsis.count('girl') + synopsis.count('female') + synopsis.count('woman')
            demographics = [d.get('name', '').lower() for d in (a.get('demographics') or [])]
            has_romance = 'romance' in genres
            
            if title_check:
                is_likely_harem = True
                harem_debug = "Known harem franchise"
            elif strong_match:
                is_likely_harem = True
                harem_debug = "Strong signal detected"
            elif harem_score >= 3:
                is_likely_harem = True
                harem_debug = f"Multiple indicators ({harem_score}): {found_indicators[:3]}"
            elif has_romance and female_count >= 2:
                is_likely_harem = True
                harem_debug = f"Romance + {female_count} female mentions"
            elif 'shounen' in demographics and has_romance and harem_score >= 1:
                is_likely_harem = True
                harem_debug = f"Shounen+Romance: {found_indicators[:2]}"
        
        # Text matching for negative filters
        full_text = f"{title} {synopsis} {genre_str} {themes_str}"
        text_match = any(neg.lower() in full_text for neg in expanded_negatives if neg and neg != 'harem')
        
        if genre_match or text_match or is_likely_harem:
            filtered_count += 1
            reason = harem_debug if is_likely_harem else ('genre' if genre_match else 'text')
            filter_reasons.append(f"{title[:30]}: {reason}")
            continue
        
        # Check if anime matches ALL required concepts
        if requires_all_concepts:
            full_content = f"{title_original} {synopsis} {genre_str} {themes_str}".lower()
            matches_all = True
            missing_concepts = []
            
            for concept_group in key_concepts:
                is_tank_group = any(t in ['tank', 'tanks', 'panzer', 'sensha'] for t in concept_group)
                
                if is_tank_group:
                    tank_keywords = ['tank', 'tanks', 'panzer', 'sensha', 'military', 'warfare', 'combat vehicle', 'armor']
                    tank_match = any(kw in full_content for kw in tank_keywords)
                    military_theme = any(t in themes_str for t in ['military', 'war', 'combat'])
                    military_genre = 'military' in genre_str
                    has_match = tank_match or military_theme or military_genre
                else:
                    has_match = any(term in full_content for term in concept_group)
                
                if not has_match:
                    matches_all = False
                    missing_concepts.append(concept_group[0])
            
            if not matches_all:
                filtered_count += 1
                filter_reasons.append(f"{title[:30]}: missing {', '.join(missing_concepts)}")
                continue
        
        if not title and not synopsis and not genre_str:
            continue
        
        # Build weighted document
        doc_title = f"{title} {title}"
        doc_genres = f"{genre_str} {genre_str} {genre_str}"
        doc_themes = f"{themes_str} {themes_str}"
        
        if is_yuri_search and 'girls love' in genres:
            doc_genres += " girls love girls love girls love girls love girls love yuri yuri yuri"
        
        if is_music_search and 'music' in genres:
            doc_genres += " music music music music music band band band instrument instrument"
        
        synopsis_boost = synopsis
        if is_cgdct_search:
            if 'slice of life' in genres:
                doc_genres += " slice of life slice of life slice of life"
            if 'comedy' in genres:
                doc_genres += " comedy comedy comedy"
            if 'girl' in synopsis or 'girls' in synopsis:
                synopsis_boost += " girls girls girls cute cute cute"
        
        title_boost = ""
        
        for term in rare_terms:
            if term in ['tank', 'tanks', 'panzer', 'sensha']:
                if term in synopsis and not any(fp in synopsis for fp in ['thinking', 'tanking', 'thankful']):
                    synopsis_boost += f" {term} {term} {term} {term} {term} {term} {term} {term} {term} {term}"
                if term in title and not any(fp in title for fp in ['thinking', 'tanking']):
                    title_boost += " ".join([term] * 15)
            else:
                if term in synopsis:
                    synopsis_boost += f" {term} {term} {term} {term} {term}"
                if term in title:
                    title_boost += f" {term} {term} {term} {term} {term} {term} {term} {term}"
        
        genre_boost = ""
        for term in rare_terms:
            if term in genre_str:
                genre_boost += f" {term} {term} {term} {term}"
        
        doc = f"{doc_title} {title_boost} {doc_genres} {genre_boost} {doc_themes} {synopsis_boost}"
        documents.append(doc)
        valid_anime.append(a)
        seen_base_titles[base_title + '_doc'] = doc
    
    if filtered_count > 0:
        reasons_summary = Counter([r.split(': ')[1] if ': ' in r else 'other' for r in filter_reasons])
        reasons_str = ', '.join([f"{count} {reason}" for reason, count in reasons_summary.most_common(5)])
        
        if needs_harem_detection or is_yuri_search or is_music_search or is_cgdct_search:
            st.caption(f"🚫 Filtered out {filtered_count} anime ({reasons_str})")
            with st.expander("🔍 See what was filtered and why"):
                for reason in filter_reasons[:30]:
                    st.text(reason)
        else:
            st.caption(f"🚫 Filtered out {filtered_count} anime ({reasons_str})")
            if list(expanded_negatives):
                st.caption(f"Also excluding: {', '.join(expanded_negatives)}")
            
            if 'missing' in reasons_str:
                with st.expander("🔍 See what was filtered and why (first 20)"):
                    for reason in filter_reasons[:20]:
                        st.text(reason)
    
    if not documents:
        return []
    
    # Clean query
    clean_query = query_lower
    for pattern in negation_patterns:
        clean_query = clean_query.replace(pattern, ' ')
    for neg in expanded_negatives:
        clean_query = clean_query.replace(neg, '')
    
    for term in rare_terms:
        clean_query += f" {term} {term} {term}"
    
    documents.append(clean_query)
    
    try:
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=1000,
            ngram_range=(1, 3),
            min_df=1,
            sublinear_tf=True,
            use_idf=True,
            norm='l2'
        )
        tfidf_matrix = vectorizer.fit_transform(documents)
        cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
        
        for idx, anime in enumerate(valid_anime):
            base_score = float(cosine_sim[0][idx]) * 100
            full_content = f"{anime.get('title', '')} {anime.get('synopsis', '')}".lower()
            exact_matches = sum(1 for term in rare_terms if term in full_content)
            boost = exact_matches * 5
            
            if is_yuri_search:
                anime_genres = [g.get('name', '').lower() for g in (anime.get('genres') or [])]
                if 'girls love' in anime_genres or 'shoujo ai' in anime_genres:
                    boost += 20
            
            if is_music_search:
                anime_genres = [g.get('name', '').lower() for g in (anime.get('genres') or [])]
                if 'music' in anime_genres:
                    boost += 20
            
            if is_cgdct_search:
                anime_genres = [g.get('name', '').lower() for g in (anime.get('genres') or [])]
                cgdct_score = 0
                if 'slice of life' in anime_genres:
                    cgdct_score += 10
                if 'comedy' in anime_genres:
                    cgdct_score += 5
                if full_content.count('girl') + full_content.count('girls') >= 2:
                    cgdct_score += 5
                boost += cgdct_score
            
            anime['vibe_score'] = round(min(base_score + boost, 100), 2)
            
    except Exception as e:
        logger.error(f"NLP processing error: {str(e)}")
        for anime in valid_anime:
            anime['vibe_score'] = 0

    return sorted(valid_anime, key=lambda x: x.get('vibe_score', 0), reverse=True)

# --- OAUTH HANDLING ---
def handle_mal_callback():
    """Processes the MAL token exchange"""
    query_params = st.query_params.to_dict()
    if 'code' in query_params and not st.session_state.mal_authenticated:
        auth_code = query_params['code']
        code_verifier = query_params.get('state')
        
        try:
            token_data = {
                'client_id': MAL_CLIENT_ID,
                'client_secret': MAL_CLIENT_SECRET,
                'code': auth_code,
                'code_verifier': code_verifier,
                'grant_type': 'authorization_code',
                'redirect_uri': REDIRECT_URI
            }
            # Make token request with proper headers and timeout
            resp = requests.post(
                "https://myanimelist.net/v1/oauth2/token",
                data=token_data,
                headers=DEFAULT_HEADERS,
                timeout=10,
                verify=True
            )
            if resp.status_code == 200:
                st.session_state.mal_authenticated = True
                st.session_state.oauth_session = OAuth2Session(MAL_CLIENT_ID, token=resp.json())
                st.query_params.clear()
                st.rerun()
            else:
                logger.error(f"Token exchange failed: {resp.status_code} - {resp.text}")
                st.error(f"❌ Auth Failed: {resp.status_code}")
        except requests.exceptions.Timeout:
            logger.error("Token request timeout")
            st.error("❌ Connection timeout. Please try again.")
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error: {str(e)}")
            st.error("❌ Connection error. Please check your internet connection.")
        except Exception as e:
            logger.error(f"Unexpected error in token exchange: {str(e)}")
            st.error(f"❌ Error: {str(e)}")

# Initialize provider selection
query_params = st.query_params.to_dict()
if 'code' in query_params:
    st.session_state.selected_provider = "MyAnimeList"
elif st.session_state.selected_provider is None:
    st.session_state.selected_provider = "Anilist"

handle_mal_callback()

# --- RECOMMENDATION ENGINE ---
def run_engine(raw_list: List[Dict], provider: str, min_score: float, vibe_q: str = "") -> List[Dict]:
    """Main recommendation engine"""
    norm = []
    seen = set()
    candidate_pool = []
    
    for item in raw_list:
        try:
            if provider == "Anilist":
                s = item.get('score', 0)
                m = item.get('media', {})
                title = m.get('title', {}).get('english') or m.get('title', {}).get('romaji', '')
                mid = m.get('idMal')
                genres = m.get('genres', [])
            elif provider == "Kitsu":
                r20 = item.get('attributes', {}).get('ratingTwenty')
                s = (float(r20)/2) if r20 else 0
                media = item.get('media_details', {}).get('attributes', {})
                title = media.get('canonicalTitle', '')
                mid = None
                genres = item.get('genres_list', [])
            elif provider == "MyAnimeList":
                s = item.get('list_status', {}).get('score', 0)
                node = item.get('node', {})
                title = node.get('title', '')
                mid = node.get('id')
                genres = []
            elif provider in ["Manual Selection", "Manual", "NLP"]:
                s = 10
                title = item.get('title', '')
                mid = item.get('mal_id')
                genres = [g['name'] for g in item.get('genres', [])]
            
            if title:
                norm.append({
                    'title': title,
                    'score': s,
                    'mal_id': mid,
                    'genres': genres
                })
        except Exception as e:
            logger.error(f"Normalization error: {str(e)}")
            continue

    p_bar = st.progress(0, text="Fetching candidates...")
    
    if provider == "NLP":
        candidate_pool = raw_list
        p_bar.empty()
    else:
        scored = [x for x in norm if x['score'] > 0]
        targets = sorted(scored, key=lambda x: x['score'], reverse=True)[:3] if scored else norm[:3]
        
        for i, anime in enumerate(targets):
            p_bar.progress((i+1)/len(targets), text=f"Analyzing {anime['title'][:30]}...")
            tid = anime['mal_id']
            
            if not tid:
                res = safe_jikan_call(jikan.search, 'anime', anime['title'], parameters={'limit':1})
                if res and res.get('data'):
                    tid = res['data'][0].get('mal_id')
            
            if tid:
                recs_data = safe_jikan_call(jikan.anime, tid, extension='recommendations')
                if recs_data and recs_data.get('data'):
                    rec_ids = [r.get('entry', {}).get('mal_id') for r in recs_data['data'][:15]]
                    rec_ids = [rid for rid in rec_ids if rid and rid not in seen][:10]
                    
                    for eid in rec_ids:
                        resp = safe_jikan_call(jikan.anime, eid)
                        if resp and resp.get('data'):
                            candidate_pool.append(resp['data'])
                            seen.add(eid)
                            time.sleep(0.1)
                    
                    time.sleep(0.5)
        
        p_bar.empty()

    if vibe_q:
        negation_patterns = ['not ', 'no ', 'without ', 'avoid ', "don't ", 'except ', 'but not ', 'minus ']
        has_negation = any(pattern in vibe_q.lower() for pattern in negation_patterns)
        
        if has_negation:
            st.info("🚫 Negative filtering active - excluding unwanted genres/themes")
        
        rare_terms, key_concepts = extract_key_concepts(vibe_q)
        if key_concepts and len(key_concepts) >= 2:
            concept_names = [group[0] for group in key_concepts]
            st.info(f"🎯 Multi-concept search: requiring ALL of [{', '.join(concept_names)}]")
        
        ranked = compute_nlp_similarity(vibe_q, candidate_pool)
        
        if provider == "NLP":
            results = [r for r in ranked if r.get('vibe_score', 0) > 0 and (r.get('score') or 0) >= min_score][:20]
            if not results:
                st.warning("⚠️ No semantic matches found. Try different keywords or lower the minimum score.")
                results = sorted(
                    [r for r in candidate_pool if (r.get('score') or 0) >= min_score],
                    key=lambda x: x.get('score') or 0,
                    reverse=True
                )[:10]
        else:
            results = [r for r in ranked if r.get('vibe_score', 0) > 1.5 and (r.get('score') or 0) >= min_score][:10]
        
        if results and ranked:
            top_score = results[0].get('vibe_score', 0) if results else 0
            st.info(f"🔍 Analyzed {len(ranked)} candidates (filtered from {len(candidate_pool)}). Top match: {top_score}%")
    else:
        results = sorted(
            candidate_pool,
            key=lambda x: x.get('score') or 0,
            reverse=True
        )[:10]

    if results:
        genre_counts = Counter()
        for r in results:
            genre_counts.update([g['name'] for g in r.get('genres', [])])
        
        if genre_counts:
            st.subheader("📊 Recommendation Vibe Profile")
            df_chart = pd.DataFrame(
                genre_counts.items(),
                columns=['Genre', 'Count']
            ).sort_values('Count', ascending=False).head(10)
            
            fig = px.pie(
                df_chart,
                values='Count',
                names='Genre',
                hole=0.5,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig, use_container_width=True)
    
    return results

# --- NLP SEARCH FUNCTION ---
def perform_nlp_search(vibe: str, min_score: float):
    """Perform NLP-based mood search"""
    progress_text = st.empty()
    progress_text.text("🔍 Analyzing your query...")
    
    collection = detect_special_collection(vibe)
    
    all_results = []
    vibe_lower = vibe.lower()
    
    is_music_search = any(term in vibe_lower for term in ['band', 'music', 'idol', 'song']) and 'cute girls' not in vibe_lower
    is_yuri_search = any(term in vibe_lower for term in ['yuri', 'girls love', 'shoujo ai', 'lesbian', 'girls doing cute girls', 'girl love'])
    is_cgdct_search = any(term in vibe_lower for term in ['cute girls doing cute things', 'cgdct', 'girls doing cute'])
    
    if collection:
        progress_text.text(f"🎯 Detected {collection.upper()} - fetching iconic titles...")
        iconic_results = fetch_iconic_anime(collection, lambda msg: progress_text.text(msg))
        
        if is_music_search and collection == 'music':
            iconic_results = [a for a in iconic_results if any(
                g.get('name', '').lower() == 'music' for g in a.get('genres', [])
            )]
        elif is_yuri_search and collection == 'yuri':
            iconic_results = [a for a in iconic_results if any(
                g.get('name', '').lower() in ['girls love', 'shoujo ai'] for g in a.get('genres', [])
            )]
        
        all_results.extend(iconic_results)
        search_terms = ICONIC_COLLECTIONS[collection]['search_terms']
    else:
        search_term = vibe.strip().lower()
        
        filler_phrases = [
            'anime with ', 'anime about ', 'anime like ',
            'show with ', 'show about ', 'show like ',
            'series with ', 'series about ', 'series like ',
            ' anime', 'anime '
        ]
        for filler in filler_phrases:
            search_term = search_term.replace(filler, ' ').strip()
        
        search_term = ' '.join(search_term.split())
        
        negation_patterns = ['not ', 'no ', 'without ', 'avoid ', "don't ", 'except ', 'but not ', 'minus ']
        for pattern in negation_patterns:
            if pattern in search_term:
                search_term = search_term.split(pattern)[0].strip()
                break
        
        search_term = search_term[:50]
        search_terms = [search_term] if search_term else ['adventure']
    
    try:
        for term_idx, term in enumerate(search_terms[:3]):
            for page in range(1, 3):
                params = {
                    'limit': 25,
                    'page': page,
                    'order_by': 'score' if page % 2 == 0 else 'popularity'
                }
                
                if is_music_search:
                    params['genres'] = '19'
                elif is_yuri_search:
                    params['genres'] = '26'
                
                progress_text.text(f"📥 Fetching '{term}' (page {page}/2)... Total: {len(all_results)}")
                
                res = safe_jikan_call(jikan.search, 'anime', term, parameters=params)
                
                if res and res.get('data'):
                    existing_ids = {a.get('mal_id') for a in all_results}
                    new_results = [a for a in res['data'] if a.get('mal_id') not in existing_ids]
                    all_results.extend(new_results)
                    time.sleep(0.4)
            
            time.sleep(0.3)
        
        progress_text.text("📊 Processing results...")
        
        if all_results:
            st.session_state.current_list = all_results
            st.session_state.provider_display_name = "NLP Search"
            st.session_state.nlp_vibe = vibe
            progress_text.empty()
            
            filter_msg = ""
            if is_music_search:
                filter_msg = " (Music genre filter applied)"
            elif is_yuri_search:
                filter_msg = " (Girls Love genre filter applied)"
            elif is_cgdct_search:
                filter_msg = " (CGDCT vibe search - no strict genre filter)"
            
            st.success(f"✅ Found {len(all_results)} candidates{filter_msg}! Scroll down and click 'Generate My Recommendations'")
            time.sleep(1)
            st.rerun()
        else:
            progress_text.empty()
            st.error("❌ No results found. Try different keywords!")
            
    except Exception as e:
        progress_text.empty()
        st.error(f"❌ Search failed: {str(e)}")
        logger.error(f"Search error: {str(e)}")

# --- MAIN UI ---
def main():
    with st.sidebar:
        st.title("🥝 Kiwi Controls")
        providers = ["Anilist", "Kitsu", "MyAnimeList", "Manual Selection", "NLP / Mood Search"]
        st.session_state.selected_provider = st.selectbox(
            "Connection Source",
            providers,
            index=providers.index(st.session_state.selected_provider)
        )
        method = st.session_state.selected_provider
        min_s = st.slider("Min. Score Filter", 0.0, 10.0, 7.5, 0.5)
        st.divider()

        if method == "MyAnimeList":
            if st.session_state.mal_authenticated:
                if st.button("📥 Load MAL Data"):
                    with st.spinner("Loading your anime list..."):
                        try:
                            resp = st.session_state.oauth_session.get(
                                "https://api.myanimelist.net/v2/users/@me/animelist",
                                params={'limit': 150, 'fields': 'list_status,title', 'status': 'completed'},
                                headers=DEFAULT_HEADERS,
                                timeout=10
                            )
                            if resp.status_code == 200:
                                st.session_state.current_list = resp.json().get('data', [])
                                st.session_state.provider_display_name = "MAL"
                                st.success(f"Loaded {len(st.session_state.current_list)} anime!")
                            else:
                                logger.error(f"Failed to load MAL data: {resp.status_code}")
                                st.error(f"Failed to load data. Status: {resp.status_code}")
                        except requests.exceptions.Timeout:
                            st.error("❌ Request timeout. Please try again.")
                        except Exception as e:
                            logger.error(f"Error loading MAL data: {str(e)}")
                            st.error(f"❌ Error loading data: {str(e)}")
            else:
                v = secrets.token_urlsafe(60)
                url = f"https://myanimelist.net/v1/oauth2/authorize?response_type=code&client_id={MAL_CLIENT_ID}&redirect_uri={REDIRECT_URI}&code_challenge={v}&code_challenge_method=plain&state={v}"
                components.html(f"""
                <button onclick="window.open('{url}', 'mal_login', 'width=600,height=700')" style="background:#2e51a2;color:white;padding:12px;border:none;border-radius:4px;cursor:pointer;font-weight:bold;font-size:16px;width:100%;">🔐 Login with MAL</button>
                """)
                st.caption("Click the button to authenticate with MyAnimeList in a popup window")

        elif method == "NLP / Mood Search":
            vibe = st.text_area(
                "What's the vibe?",
                value=st.session_state.nlp_vibe,
                height=100,
                placeholder="e.g., 'pokemon', 'digimon', 'cute girls doing cute things', 'vampire anime'"
            )
            st.caption("💡 Try: 'pokemon', 'digimon', 'vampire anime', 'band anime', 'cute girls doing cute things'")
            st.caption("🚫 Use negation: 'isekai but not harem'")
            
            if st.session_state.current_list and st.session_state.provider_display_name == "NLP Search":
                st.info(f"📚 Loaded: {len(st.session_state.current_list)} candidates")
                if st.button("🔄 Clear Cache"):
                    st.session_state.current_list = []
                    st.session_state.nlp_vibe = ""
                    st.rerun()
            
            if st.button("🚀 Search by Vibe", type="primary"):
                if not vibe.strip():
                    st.error("Please enter a vibe/mood to search for!")
                else:
                    perform_nlp_search(vibe, min_s)

        elif method == "Manual Selection":
            q = st.text_input("Search anime title")
            if q:
                with st.spinner("Searching..."):
                    res = safe_jikan_call(jikan.search, 'anime', q)
                    if res and res.get('data'):
                        hits = res['data'][:5]
                        sel = st.selectbox("Results:", hits, format_func=lambda x: x.get('title', 'Unknown'))
                        if st.button("➕ Add to Favorites") and len(st.session_state.manual_favs) < 5:
                            st.session_state.manual_favs.append(sel)
                            st.success(f"Added {sel.get('title')}!")
                            st.rerun()
            
            if st.session_state.manual_favs:
                st.write("**Your Favorites:**")
                for idx, f in enumerate(st.session_state.manual_favs):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write(f"✅ {f.get('title')}")
                    with col2:
                        if st.button("❌", key=f"remove_{idx}"):
                            st.session_state.manual_favs.pop(idx)
                            st.rerun()
                
                if st.button("🚀 Process Selection"):
                    st.session_state.current_list = st.session_state.manual_favs
                    st.session_state.provider_display_name = "Manual"
                    st.success("Ready to generate recommendations!")

        else:
            user = st.text_input(f"{method} Username")
            if user and st.button("📥 Load"):
                with st.spinner(f"Loading {method} data..."):
                    try:
                        if method == "Anilist":
                            q = 'query($u:String){MediaListCollection(userName:$u,type:ANIME){lists{entries{score(format:POINT_10)media{idMal genres title{romaji english}}}}}}'
                            r = requests.post(
                                ANILIST_URL,
                                json={'query': q, 'variables': {'u': user}}
                            ).json()
                            
                            if 'data' in r and r['data']['MediaListCollection']:
                                st.session_state.current_list = [
                                    e for l in r['data']['MediaListCollection']['lists']
                                    for e in l['entries']
                                ]
                                st.session_state.provider_display_name = method
                                st.success(f"Loaded {len(st.session_state.current_list)} anime!")
                            else:
                                st.error("User not found or no anime list!")
                        
                        else:
                            u_resp = requests.get(f"{KITSU_REST_URL}/users?filter[slug]={user}")
                            if u_resp.status_code == 200 and u_resp.json().get('data'):
                                u_id = u_resp.json()['data'][0]['id']
                                url = f"{KITSU_REST_URL}/library-entries?filter[user_id]={u_id}&filter[kind]=anime&include=anime,anime.categories&page[limit]=100"
                                resp = requests.get(url).json()
                                inc = {f"{i['type']}_{i['id']}": i for i in resp.get('included', [])}
                                st.session_state.current_list = []
                                
                                for e in resp.get('data', []):
                                    ref = e['relationships']['anime']['data']
                                    if f"anime_{ref['id']}" in inc:
                                        media = inc[f"anime_{ref['id']}"]
                                        e['media_details'] = media
                                        c_refs = media['relationships']['categories']['data']
                                        e['genres_list'] = [
                                            inc[f"categories_{c['id']}"]['attributes']['title']
                                            for c in c_refs
                                            if f"categories_{c['id']}" in inc
                                        ]
                                        st.session_state.current_list.append(e)
                                
                                st.session_state.provider_display_name = method
                                st.success(f"Loaded {len(st.session_state.current_list)} anime!")
                            else:
                                st.error("User not found!")
                    except Exception as e:
                        st.error(f"Error loading data: {str(e)}")
                        logger.error(f"Load error: {str(e)}")

    st.title("🥝 Kiwi's Anime Recommendation System")
    st.markdown("*Your AI-powered anime recommendation engine*")
    
    if st.session_state.current_list:
        st.success(f"✅ Data ready from **{st.session_state.provider_display_name}** ({len(st.session_state.current_list)} entries)")
        
        q = st.session_state.nlp_vibe if method == "NLP / Mood Search" else st.text_input("💬 Apply semantic vibe filter (optional):")
        
        if st.button("✨ Generate My Recommendations", type="primary"):
            with st.spinner("🔮 Analyzing your taste profile..."):
                recs = run_engine(
                    st.session_state.current_list,
                    method if method != "NLP / Mood Search" else "NLP",
                    min_s,
                    q
                )
                
                if not recs:
                    st.warning("🤔 No matches found. Try broadening your vibe description or lowering the minimum score!")
                else:
                    st.success(f"🎉 Found {len(recs)} perfect matches!")
                    st.divider()
                    
                    for idx, a in enumerate(recs):
                        c1, c2 = st.columns([1, 4])
                        with c1:
                            if a.get('images', {}).get('jpg', {}).get('large_image_url'):
                                st.image(a['images']['jpg']['large_image_url'])
                        with c2:
                            match = f"🎯 Vibe Match: **{a['vibe_score']}%**" if 'vibe_score' in a else ""
                            st.subheader(f"#{idx+1} {a.get('title', 'Unknown')} {match}")
                            
                            score = a.get('score', 'N/A')
                            genres = ', '.join([g['name'] for g in a.get('genres', [])])
                            st.write(f"⭐ MAL Score: **{score}** | 🎭 Genres: {genres}")
                            
                            if a.get('episodes'):
                                st.write(f"📺 Episodes: {a['episodes']}")
                            
                            with st.expander("📖 Read Synopsis"):
                                st.write(a.get('synopsis', 'No synopsis available.'))
                            
                            if a.get('url'):
                                st.markdown(f"[View on MyAnimeList]({a['url']})")
                        
                        st.divider()
    else:
        st.info("👈 Select a data source from the sidebar to get started!")
        st.markdown("""
        ### How to use Kiwi's Anime Recommendation System:
        1. **Choose your data source** (Anilist, Kitsu, MAL, or Manual)
        2. **Load your anime list** or search by vibe
        3. **Add a semantic filter** to fine-tune results (optional)
        4. **Generate recommendations** and discover your next favorite anime!
        
        ---
        
        **Pro tip:** Use NLP/Mood Search to find anime based purely on vibes like:
        - "pokemon" or "digimon" - Find monster-catching adventures
        - "cute girls doing cute things" - Relaxing slice-of-life
        - "vampire anime" - Dark supernatural tales
        - "band anime with comedy" - Musical comedies
        - "isekai but not harem" - Fantasy adventures without romance
        """)

if __name__ == "__main__":
    main()
