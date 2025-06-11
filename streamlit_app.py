import streamlit as st
import sqlite3
import json
from sentence_transformers import SentenceTransformer
import time
import pandas as pd
import sqlite_vss
import re

# --- Configuration & Caching (Same as before) ---
DATABASE_PATH = 'prospects.db'
# IMPORTANT: The model name here is only for the app to know what's in the DB.
# The actual embedding generation happens in the prep script.
MODEL_NAME = 'all-mpnet-base-v2' 

@st.cache_resource
def load_model():
    # This now loads the larger, better model
    return SentenceTransformer(MODEL_NAME)

# --- Query Parsing (Same as before) ---
@st.cache_data
def parse_hybrid_query(query: str):
    filter_pattern = r'\[(.*?):(.*?)\]'
    filters = re.findall(filter_pattern, query)
    semantic_query = re.sub(filter_pattern, '', query).strip()
    filter_dict = {key.strip().lower(): value.strip() for key, value in filters} # Use lowercase keys
    return semantic_query, filter_dict

# --- THE UPGRADED HYBRID SEARCH FUNCTION ---
@st.cache_data
def search_players_hybrid(semantic_query: str, filters: dict, limit: int = 20) -> (list, float):
    start_time = time.time()
    model = load_model()
    
    conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vss.load(conn)
    cursor = conn.cursor()

    candidate_rowids = None

    # Stage 1: Structured Filters (Same as before)
    if filters:
        # NOTE: Ensure your filter keys are lowercase in the dictionary now
        where_clauses = [f"LOWER({key}) LIKE ?" for key in filters.keys()]
        params = [f"%{value}%" for value in filters.values()]
        filter_query = f"SELECT rowid FROM players WHERE {' AND '.join(where_clauses)}"
        try:
            candidate_results = cursor.execute(filter_query, params).fetchall()
            candidate_rowids = {row['rowid'] for row in candidate_results}
            if not candidate_rowids:
                conn.close()
                return [], time.time() - start_time
        except sqlite3.OperationalError:
            st.error(f"Filter Error: One of your filter keys is not a valid column.")
            conn.close()
            return [], time.time() - start_time

    # Stage 2: Semantic Search (Get a larger pool of 100 candidates)
    query_vector_json = json.dumps(model.encode(semantic_query).tolist())
    vss_query = "SELECT rowid, distance FROM vss_players WHERE vss_search(player_embedding, vss_search_params(?, 100))"
    vss_results_raw = cursor.execute(vss_query, (query_vector_json,)).fetchall()
    
    # Create a dictionary of {rowid: distance} for easy lookup
    vss_results_map = {row['rowid']: row['distance'] for row in vss_results_raw}
    
    # Intersect VSS results with structured candidates
    if candidate_rowids is not None:
        intersected_ids = candidate_rowids.intersection(vss_results_map.keys())
    else:
        intersected_ids = vss_results_map.keys()
        
    if not intersected_ids:
        conn.close()
        return [], time.time() - start_time

    # Retrieve full data for the candidate players
    placeholders = ','.join('?' for _ in intersected_ids)
    players_query = f"SELECT rowid, * FROM players WHERE rowid IN ({placeholders})"
    full_player_data = cursor.execute(players_query, list(intersected_ids)).fetchall()
    conn.close()

    # --- Step 3: HYBRID RERANKING LOGIC ---
    reranked_results = []
    for player_row in full_player_data:
        player = dict(player_row) # Convert to a standard dictionary
        rowid = player['rowid']
        
        # Calculate Semantic Score (higher is better)
        # We invert the distance and add 1 to avoid division by zero
        semantic_score = 1.0 / (1.0 + vss_results_map[rowid])

        # Calculate Draft Score (higher is better)
        overall_pick = player.get('overall')
        if pd.isna(overall_pick):
            draft_score = 1.0 / 500.0 # Penalize any undrafted players heavily
        else:
            draft_score = 1.0 / (1.0 + overall_pick)

        # Define weights (TUNE THESE to change behavior)
        # We give more weight to draft position to solve the "generational talent" problem
        w_semantic = 0.4 
        w_draft = 0.6
        
        hybrid_score = (w_semantic * semantic_score) + (w_draft * draft_score)
        
        reranked_results.append({'player': player, 'score': hybrid_score})

    # Sort the final list by the new hybrid score, highest first
    final_sorted_list = sorted(reranked_results, key=lambda x: x['score'], reverse=True)
    
    # Return only the top 'limit' players from the reranked list
    final_players = [item['player'] for item in final_sorted_list][:limit]

    search_duration = time.time() - start_time
    return final_players, search_duration

# --- Streamlit User Interface (Definitive Final Version) ---
st.set_page_config(page_title="NFL Prospect Search (API-Powered)", page_icon="🏈", layout="wide")
st.title("🏈 NFL Prospect Search (API-Powered)")
st.markdown("Search from 2014-2025 using semantic queries and structured filters like `[position:QB]` or `[school_name:Alabama]`")
query = st.text_input("Search for a prospect", placeholder="e.g., 'elusive running back with good vision'")

if query:
    semantic_part, filters = parse_hybrid_query(query)
    if not semantic_part:
        st.warning("Please provide a text query.")
    else:
        results, duration = search_players_hybrid(semantic_part, filters)
        st.info(f"Found {len(results)} results in {duration:.2f} seconds.")
        if not results:
            st.warning("No matching players found.")
        else:
            # Helper function to render a player card correctly
            def render_player_card(player_data):
                with st.container(border=True):
                    # --- Line 1: Name, Position, and Year ---
                    # Use the 'year' field from the API. This fixes the "Undrafted/Future" bug.
                    name_text = f"**{player_data.get('player_name', 'N/A')}**"
                    pos_text = f"({player_data.get('position', 'N/A')})"
                    year_text = f"Prospect Year: **{int(player_data.get('year', ''))}**" if pd.notna(player_data.get('year')) else ""
                    st.markdown(f"{name_text} {pos_text} | {year_text}")
                    
                    # --- Line 2: School and Grade ---
                    # Add the 'grade' field as requested.
                    grade = player_data.get('grade')
                    grade_text = f"Grade: **{grade:.2f}**" if pd.notna(grade) else ""
                    school_text = f"School: {player_data.get('school_name', 'N/A')}"
                    st.caption(f"{school_text} | {grade_text}")
                    
                    # --- Line 3: Draft Projection ---
                    # Add the 'draft_projection' field as requested.
                    projection_text = player_data.get('draft_projection')
                    if pd.notna(projection_text):
                        st.markdown(f"**Projection:** {projection_text}")

                    # Expander for analysis
                    with st.expander("Show Scouting Analysis (Strengths & Weaknesses)"):
                        st.write(player_data.get('analysis_text', 'No analysis available.'))

            # Render cards in columns
            for i in range(0, len(results), 2):
                col1, col2 = st.columns(2)
                if i < len(results):
                    with col1:
                        render_player_card(results[i])
                if i + 1 < len(results):
                    with col2:
                        render_player_card(results[i+1])
else:
    st.info("Enter a query above to start your search.")