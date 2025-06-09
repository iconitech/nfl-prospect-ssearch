import streamlit as st
import sqlite3
import json
from sentence_transformers import SentenceTransformer
import time
import pandas as pd
import sqlite_vss

# --- Configuration ---
DATABASE_PATH = 'prospects.db'
MODEL_NAME = 'all-MiniLM-L6-v2'

# --- Caching ---
# Caching is Streamlit's superpower. It prevents re-loading the model and
# re-running searches for the same query, making the app feel instantaneous.

@st.cache_resource
def load_model():
    """Loads the SentenceTransformer model and caches it."""
    return SentenceTransformer(MODEL_NAME)

@st.cache_data
def search_players(query_text: str, limit: int = 20) -> (list, float):
    """
    Performs semantic search on the database.
    This function is cached so if the same query is entered again,
    the result is returned instantly without hitting the database.
    """
    start_time = time.time()
    model = load_model() # Retrieve the cached model
    query_vector_json = json.dumps(model.encode(query_text).tolist())

    conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    
    # This is the platform-agnostic way to load the VSS extension.
    # It's much more robust for deployment than hardcoding the file path.
    sqlite_vss.load(conn)
    
    cursor = conn.cursor()

    # This is the backwards-compatible VSS query for older SQLite versions
    vss_query = """
        SELECT rowid, distance
        FROM vss_players
        WHERE vss_search(
            player_embedding,
            vss_search_params(?, ?) 
        )
    """
    try:
        # Note the new parameters: we pass the vector and the limit INSIDE the query params
        results = cursor.execute(vss_query, (query_vector_json, limit)).fetchall()
        # The returned results are already limited and sorted by distance, so we don't need ORDER BY or LIMIT
        matched_rowids = [row['rowid'] for row in results]
        
        if not matched_rowids:
            return [], 0.0

        placeholders = ','.join('?' for _ in matched_rowids)
        players_query = f"SELECT * FROM players WHERE rowid IN ({placeholders})"
        
        full_results_unordered = cursor.execute(players_query, matched_rowids).fetchall()
        # Create a mapping to re-sort results based on VSS distance
        results_map = {dict(row)['rowid']: dict(row) for row in full_results_unordered}
        full_results = [results_map[rowid] for rowid in matched_rowids if rowid in results_map]

    finally:
        conn.close()
    
    search_duration = time.time() - start_time
    return full_results, search_duration

# --- Streamlit User Interface ---

# Page Configuration (set this as the first st command)
st.set_page_config(
    page_title="NFL Prospect Semantic Search",
    page_icon="🏈",
    layout="wide"
)

st.title("🏈 NFL Prospect Semantic Search")

# Search Bar
query = st.text_input(
    "Search for a prospect's scouting report",
    placeholder="e.g., 'big-armed QB with accuracy issues'",
    help="Describe the type of player you're looking for."
)

# Perform search when query is entered
if query:
    results, duration = search_players(query)
    
    st.info(f"Found {len(results)} results in {duration:.2f} seconds.")
    
    if not results:
        st.warning("No matching players found.")
    else:
        # Display results in columns for a cleaner look
        for i in range(0, len(results), 2):
            col1, col2 = st.columns(2)
            
            # --- Player in Column 1 ---
            with col1:
                player = results[i]
                with st.container(border=True):
                    name_text = f"**{player.get('player_name', 'N/A')}** ({player.get('position', 'N/A')})"
                    
                    draft_text = "Undrafted or Future Prospect"
                    if not pd.isna(player.get('round')):
                        draft_text = f"Drafted: R{int(player.get('round'))} P{int(player.get('overall'))} by {player.get('team', 'N/A')}"
                    
                    st.markdown(f"{name_text} | {draft_text}")
                    st.caption(f"School: {player.get('school_name', 'N/A')}")
                    
                    with st.expander("Show Scouting Analysis"):
                        st.write(player.get('analysis_text', 'No analysis available.'))
            
            # --- Player in Column 2 (if exists) ---
            if i + 1 < len(results):
                with col2:
                    player = results[i+1]
                    with st.container(border=True):
                        name_text = f"**{player.get('player_name', 'N/A')}** ({player.get('position', 'N/A')})"
                    
                        draft_text = "Undrafted or Future Prospect"
                        if not pd.isna(player.get('round')):
                            draft_text = f"Drafted: R{int(player.get('round'))} P{int(player.get('overall'))} by {player.get('team', 'N/A')}"
                        
                        st.markdown(f"{name_text} | {draft_text}")
                        st.caption(f"School: {player.get('school_name', 'N/A')}")
                        
                        with st.expander("Show Scouting Analysis"):
                            st.write(player.get('analysis_text', 'No analysis available.'))

else:
    st.info("Enter a search query above to see results.")