import streamlit as st
import pandas as pd
import pyarrow.feather as feather
import pyarrow as pa
import pyarrow.dataset as ds
import json
import os
import re
import sys # Added for PyInstaller path handling
from pathlib import Path
from collections import defaultdict
import time # For potential debugging/timing

SEGMENTS_PATH = Path("news-segments-small.feather")
SCHEMA_PATH = Path("new-schema.json")
TOPICS_SUBSET_PATH = Path("topics-subset.feather")
OUTPUT_PATH = Path("manual-news.jsonl")

# --- Helper Functions ---

def sanitize_for_col_name(text: str) -> str:
    """Sanitizes a category name for use in a t_* column."""
    txt = text.lower()
    # Replace slashes, punctuation, and whitespace with underscores
    txt = re.sub(r'[?.,;:!/]+', '_', txt)
    txt = re.sub(r'\s+', '_', txt.strip())
    # Remove potential leading/trailing underscores and multiple underscores
    txt = re.sub(r'_+', '_', txt).strip('_')
    return f"t_{txt}"

def create_schema_mappings(schema_data: list) -> tuple[list[str], dict, dict, dict]:
    """
    Parses the schema to create mappings between category, category_name, and t_* column names.

    Returns:
        tuple: (
            all_categories: List of displayable category strings, sorted.
            cat_to_name_map: Mapping from display category -> category_name.
            name_to_cat_map: Mapping from category_name -> display category.
            t_col_to_cat_map: Mapping from t_* column name -> display category.
        )
    """
    all_categories_set = set()
    cat_to_name_map = {}
    name_to_cat_map = {}
    t_col_to_cat_map = {}

    def process_category(category: dict):
        display_cat = category.get('category')
        cat_name = category.get('category_name')

        if display_cat and cat_name:
            all_categories_set.add(display_cat)
            cat_to_name_map[display_cat] = cat_name
            name_to_cat_map[cat_name] = display_cat
            t_col = sanitize_for_col_name(cat_name)
            t_col_to_cat_map[t_col] = display_cat

        # Recurse children
        children = category.get('children', [])
        for child in children:
            process_category(child)

    for top_level_category in schema_data:
        process_category(top_level_category)

    all_categories = sorted(list(all_categories_set))
    return all_categories, cat_to_name_map, name_to_cat_map, t_col_to_cat_map

def load_schema(schema_path: Path) -> list:
    """Loads the JSON schema file."""
    try:
        with open(schema_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Schema file not found at {schema_path}")
        return []
    except json.JSONDecodeError:
        st.error(f"Error decoding JSON from {schema_path}")
        return []

def load_data(segments_path: Path, topics_subset_path: Path) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Loads the segments and topics Feather files."""
    segments_df = None
    topics_df = None

    # Step 1: Load the segments file first
    if not segments_path.exists():
        st.error(f"Segments file not found at {segments_path}")
    else:
        segments_df = feather.read_feather(segments_path)
    
    # Step 2: Load the topics file
    if not topics_subset_path.exists():
        st.error(f"Topics file not found at {topics_subset_path}")
    else:
        topics_df = feather.read_feather(topics_subset_path)

    
    segments_df = segments_df[segments_df['id'].isin(topics_df['id'])]
    # shuffle segments_df
    segments_df = segments_df.sample(frac=1).reset_index(drop=True)
    return segments_df, topics_df

# --- Streamlit App ---

st.set_page_config(layout="wide", page_title="News Segment Classifier")
st.markdown("""
<style>
/* Global button styling */
div.stButton > button {
   width: 100% !important;
   min-height: 45px !important;
}
div.stForm button {
    width: 100% !important;
    min-height: 45px !important;
}

/* Global multiselect styling (retained from previous adjustments) */
div[data-testid="stMultiSelect"] {
    min-height: 50px;
}

.stMultiSelect div[role="listbox"] span {
  font-size: 16px !important;
  padding: 8px 6px !important;
}

.stMultiSelect div[role="listbox"] {
  min-width: 400px !important;
}

.stMultiSelect div[data-baseweb="tag"] {
  height: auto !important;
  min-height: 30px !important;
  padding: 4px 10px !important;
  font-size: 16px !important;
  max-width: 800px !important;
  white-space: normal !important;
}

</style>
""", unsafe_allow_html=True)

st.title("Manual News Segment Classification")

# --- Initialization and Data Loading (run once using session state) ---
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.error_occurred = False
    st.session_state.session_classifications = [] # Initialize list for session classifications

# Attempt initialization only if not already done or if an error occurred previously
if not st.session_state.initialized or st.session_state.error_occurred:
    st.info("Initializing and loading data... Please wait.") # Use st.info or st.spinner
    start_time = time.time()
    init_step_start = time.time()

    with st.spinner("Loading schema..."):
        st.session_state.error_occurred = False # Reset error flag
        st.session_state.schema_data = load_schema(SCHEMA_PATH)
        if not st.session_state.schema_data:
            st.session_state.error_occurred = True
        print(f"Time loading schema: {time.time() - init_step_start:.2f}s")

    if not st.session_state.error_occurred:
        init_step_start = time.time()
        with st.spinner("Creating schema mappings..."):
            (st.session_state.all_categories,
             st.session_state.cat_to_name_map,
             st.session_state.name_to_cat_map,
             st.session_state.t_col_to_cat_map) = create_schema_mappings(st.session_state.schema_data)
            if not st.session_state.all_categories:
                 st.error("Failed to parse categories from schema.")
                 st.session_state.error_occurred = True
            print(f"Time creating schema mappings: {time.time() - init_step_start:.2f}s")

    if not st.session_state.error_occurred:
        init_step_start = time.time()
        with st.spinner("Loading data files (segments and topics)..."):
            segments_df, topics_df = load_data(SEGMENTS_PATH, TOPICS_SUBSET_PATH)
            if segments_df is None or topics_df is None:
                st.error("Failed to load necessary data files. Please check paths and file integrity in console/logs.")
                st.session_state.error_occurred = True
            print(f"Time loading data files: {time.time() - init_step_start:.2f}s")

    if not st.session_state.error_occurred:
        init_step_start = time.time()
        with st.spinner("Filtering segments..."):
            # Filter segments to include only those present in topics_df
            classified_ids = set(topics_df['id'].unique())
            initial_segment_count = len(segments_df)
            segments_df_filtered = segments_df[segments_df['id'].isin(classified_ids)].reset_index(drop=True)
            filtered_segment_count = len(segments_df_filtered)

            if filtered_segment_count == 0:
                st.warning(f"No segments found in {SEGMENTS_PATH.name} that have corresponding classifications in {TOPICS_SUBSET_PATH.name}.")
                st.session_state.error_occurred = True
            # else: # No need to log this every time if filtering is fast
            #      if filtered_segment_count < initial_segment_count:
            #           st.info(f"Filtered segments: Kept {filtered_segment_count} out of {initial_segment_count} segments based on presence in topics file.")
            st.session_state.segments_df = segments_df_filtered
            print(f"Time filtering segments: {time.time() - init_step_start:.2f}s")


    if not st.session_state.error_occurred:
        init_step_start = time.time()
        with st.spinner("Processing topics data (grouping)..."):
            # Prepare topics_df for grouping: Select relevant columns, handle missing, ensure types
            topic_cols_from_schema = list(st.session_state.t_col_to_cat_map.keys())
            # Identify which of these columns actually exist in topics_df
            valid_topic_cols_in_df = [col for col in topic_cols_from_schema if col in topics_df.columns]
            missing_cols = set(topic_cols_from_schema) - set(valid_topic_cols_in_df)
            # if missing_cols: # Be less verbose, only warn if critical
            #     st.info(f"Note: The following expected topic columns were not found in {TOPICS_SUBSET_PATH.name} and will be treated as False: {', '.join(sorted(list(missing_cols)))}")

            required_cols = ['id', 'run'] + valid_topic_cols_in_df
            topics_subset = topics_df[required_cols].copy() # Work on a copy

            # Ensure boolean type for valid topic columns, fill NaNs if any occurred before loading
            bool_conversion_dict = {}
            for col in valid_topic_cols_in_df:
                 # Convert non-boolean types or fill NaNs before converting
                 if not pd.api.types.is_bool_dtype(topics_subset[col]):
                      # This fillna + astype could be slow if df is large and many cols need conversion
                      topics_subset[col] = topics_subset[col].fillna(False).astype(bool)
                 bool_conversion_dict[col] = bool # Final type confirmation

            topics_subset = topics_subset.astype(bool_conversion_dict)

            # Add missing columns as False
            for col in missing_cols:
                 topics_subset[col] = False

            st.session_state.topics_df_grouped = topics_subset.groupby('id')
            # print("Debug: Grouped DataFrame Info:") # Debug print
            # print(st.session_state.topics_df_grouped.size()) # Debug print
            print(f"Time processing topics data (columns, types, group): {time.time() - init_step_start:.2f}s")

        init_step_start = time.time()
        with st.spinner("Reading progress file..."):
            # --- Final Initialization Steps ---
            st.session_state.all_segment_ids = st.session_state.segments_df['id'].tolist()
            st.session_state.total_segments = len(st.session_state.all_segment_ids)
            # Load progress if output file exists
            processed_ids = set()
            if OUTPUT_PATH.exists():
                try:
                    with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                data = json.loads(line)
                                if 'id' in data:
                                    processed_ids.add(str(data['id'])) # Ensure comparison with string IDs
                            except json.JSONDecodeError:
                                st.warning(f"Skipping invalid line in {OUTPUT_PATH.name}")
                except Exception as e:
                    st.error(f"Error reading progress file {OUTPUT_PATH.name}: {e}")

            # Find the first unprocessed segment index
            start_index = 0
            for idx, segment_id in enumerate(st.session_state.all_segment_ids):
                 if segment_id not in processed_ids:
                      start_index = idx
                      break
            else: # If loop completes without break, all are processed
                 if st.session_state.total_segments > 0: # Check if there were segments to begin with
                     start_index = st.session_state.total_segments

            st.session_state.current_index = start_index
            st.session_state.output_path = OUTPUT_PATH
            print(f"Time reading progress file and setting index: {time.time() - init_step_start:.2f}s")

            st.session_state.initialized = True
            total_init_time = time.time() - start_time
            print(f"Total Initialization Time: {total_init_time:.2f}s")
            st.success(f"Initialization complete ({total_init_time:.2f}s).")
            # time.sleep(0.5) # Brief pause optional
            st.rerun() # Rerun to clear initialization messages and spinner

# --- Main Application Logic ---

# Stop if initialization failed
if not st.session_state.initialized:
     st.warning("App not initialized. Please refresh.")
     st.stop()

# --- Session Download Button ---
# Offer download only if classifications have been made in this session
if st.session_state.session_classifications:
    try:
        # Format session classifications as a JSONL string
        session_jsonl_data = "\n".join(json.dumps(record) for record in st.session_state.session_classifications)
        st.download_button(
            label=f"Download Session Classifications ({len(st.session_state.session_classifications)} items)",
            data=session_jsonl_data,
            file_name="session-classifications.jsonl", # Changed filename
            mime="application/jsonl",
            key="download_session_button" # Changed key
        )
    except Exception as e:
        st.warning(f"Could not prepare session classifications for download: {e}")
else:
    # Optionally, show a disabled button or just info text
    st.info("No classifications made in this session yet. The download button will appear after you save the first one.")
# --- End Session Download Button ---

if st.session_state.current_index >= st.session_state.total_segments:
    st.success("🎉 All segments have been classified! 🎉")
    st.balloons()
    st.stop()

# Helper functions for classification
def get_run_classifications(segment_id: str, topics_df_grouped, t_col_to_cat_map: dict) -> dict[str, list[str]]:
    run_classifications = defaultdict(list)
    try:
        segment_group = topics_df_grouped.get_group(segment_id)
        for _, row in segment_group.iterrows():
            run_name = row['run']
            for t_col, cat in t_col_to_cat_map.items():
                if t_col in row and row[t_col]: # Check if column exists and is True
                    run_classifications[run_name].append(cat)
            run_classifications[run_name].sort() # Sort for consistent display
    except KeyError:
        # No classifications found for this segment ID in the grouped data
        pass # Return empty defaultdict
    except Exception as e:
        st.warning(f"Error getting run classifications for {segment_id}: {e}")
    return dict(run_classifications)

def calculate_initial_final_topics(segment_id: str, topics_df_grouped, t_col_to_cat_map: dict) -> tuple[list[str], list[str]]:
    majority_categories = []
    minority_categories = []
    try:
        segment_group = topics_df_grouped.get_group(segment_id)
        num_runs = len(segment_group)
        if num_runs == 0:
            return [], []

        topic_counts = defaultdict(int)
        for t_col, cat in t_col_to_cat_map.items():
             if t_col in segment_group.columns: # Check column exists before summing
                 count = segment_group[t_col].sum()
                 if count > 0:
                     topic_counts[cat] = count

        majority_threshold = num_runs / 2.0

        for cat, count in topic_counts.items():
            if count > majority_threshold:
                majority_categories.append(cat)
            else: # count > 0 and count <= majority_threshold
                minority_categories.append(cat)

        majority_categories.sort()
        minority_categories.sort()

    except KeyError:
        # No classifications found for this segment ID
         pass
    except Exception as e:
        st.warning(f"Error calculating initial topics for {segment_id}: {e}")

    return majority_categories, minority_categories


def save_classification(segment_id: str, selected_categories: list[str], cat_to_name_map: dict, output_path: Path):
    """Saves the selected classification to the output JSONL file."""
    try:
        # Convert display categories back to category_names for saving
        final_topic_names = sorted([cat_to_name_map[cat] for cat in selected_categories if cat in cat_to_name_map])

        output_record = {
            "id": segment_id,
            "final_topics": final_topic_names
        }
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(output_record) + '\n')
        # Optional: Log success
        # print(f"Saved classification for {segment_id}")
    except Exception as e:
        st.error(f"Error saving classification for {segment_id}: {e}")

def organize_categories_hierarchically(schema_data, current_path=[], order_dict=None, current_order=0, parent_map=None):
    """Recursively organize categories into a hierarchical structure and maintain order."""
    result = {}
    if order_dict is None:
        order_dict = {}
    if parent_map is None:
        parent_map = {}
    
    for category in schema_data:
        cat_name = category.get('category')
        if cat_name:
            new_path = current_path + [cat_name]
            path_key = " > ".join(new_path)
            result[path_key] = cat_name
            # Store the original order from schema
            order_dict[cat_name] = current_order
            current_order += 1
            
            # Store parent relationship (direct parent only)
            if current_path:
                parent_map[cat_name] = current_path[-1]
            
        # Process children recursively
        children = category.get('children', [])
        if children:
            child_results, order_dict, current_order, parent_map = organize_categories_hierarchically(
                children, 
                current_path + ([cat_name] if cat_name else []),
                order_dict,
                current_order,
                parent_map
            )
            result.update(child_results)
            
    return result, order_dict, current_order, parent_map

# Function to get all parents of a category including ancestors
def get_all_parents(category, parent_map):
    """Get all parent categories including ancestors."""
    parents = []
    current = category
    while current in parent_map:
        parent = parent_map[current]
        parents.append(parent)
        current = parent
    return parents

# Function to extract the last part of a hierarchical path for display
def get_category_display_name(category, hierarchical_categories):
    """Get just the category name without the parent path for display."""
    for path, cat in hierarchical_categories.items():
        if cat == category:
            # Extract just the last part of the path (the category itself)
            return path.split(" > ")[-1]
    return category  # Fallback to the original name if not found

# Get current segment details
current_segment_id = st.session_state.all_segment_ids[st.session_state.current_index]
segment_data = st.session_state.segments_df.iloc[st.session_state.current_index]
segment_text = segment_data.get('text', 'Text not found')

st.header(f"Segment {st.session_state.current_index + 1} of {st.session_state.total_segments}")

# --- Jump to ID functionality ---
col1, col2 = st.columns([3, 1]) # Create columns for layout
with col1:
    jump_id_input = st.text_input(
        "Enter Segment ID to jump to:",
        key="jump_id_input",
        placeholder="Paste or type segment ID here"
    )
with col2:
    # Add some vertical space to align button better with text input
    st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
    jump_button = st.button("Jump to ID", key="jump_button")

if jump_button:
    entered_id = jump_id_input.strip()
    if entered_id:
        try:
            # Find the index of the entered ID
            target_index = st.session_state.all_segment_ids.index(entered_id)
            st.session_state.current_index = target_index
            st.rerun() # Rerun to update the display to the new segment
        except ValueError:
            st.error(f"Segment ID '{entered_id}' not found.")
        except Exception as e:
            st.error(f"An error occurred while trying to jump: {e}")
    else:
        st.warning("Please enter a Segment ID to jump.")
# --- End Jump to ID ---

st.subheader(f"ID: `{current_segment_id}`")

# Display the segment text more prominently with a larger font
st.markdown(f"""
<div style="background-color: white; padding: 15px; border-radius: 5px; border: 1px solid #ddd;">
    <p style="font-size: 18px; color: black;">{segment_text}</p>
</div>
""", unsafe_allow_html=True)

# Hard-code use_previous as a boolean instead of using a checkbox
use_previous = False

# --- Display Run Classifications only if use_previous is True ---
if use_previous:
    st.subheader("Run Classifications")
    run_classifications = get_run_classifications(
        current_segment_id,
        st.session_state.topics_df_grouped,
        st.session_state.t_col_to_cat_map
    )

    if run_classifications:
        # Use columns for potentially better layout if many runs
        num_runs = len(run_classifications)
        cols = st.columns(min(num_runs, 4)) # Max 4 columns for readability
        run_names_sorted = sorted(run_classifications.keys())
        for i, run_name in enumerate(run_names_sorted):
            with cols[i % len(cols)]:
                st.markdown(f"**{run_name}:**")
                if run_classifications[run_name]:
                    # Display as a list
                    st.markdown("\n".join(f"- {cat}" for cat in run_classifications[run_name]))
                else:
                    st.markdown("_No topics found_")
    else:
        st.info("No previous classifications found for this segment in the topics file.")

# --- Classification Selection ---
# No title to keep it clean

majority_cats, minority_cats = calculate_initial_final_topics(
    current_segment_id,
    st.session_state.topics_df_grouped,
    st.session_state.t_col_to_cat_map
)

# Set default value for multiselect based on use_previous setting
default_categories = majority_cats if use_previous else []

# Create hierarchical category display with order information
hierarchical_categories, category_order, _, parent_map = organize_categories_hierarchically(st.session_state.schema_data)

# Sort all_categories by their order in the schema instead of alphabetically
sorted_categories = sorted(st.session_state.all_categories, key=lambda cat: category_order.get(cat, 999999))

# For selection, show full hierarchical paths in dropdown but only category names for selected items
selected_final_categories = st.multiselect(
    "Select categories:",
    options=sorted_categories,
    default=default_categories,
    format_func=lambda x: get_category_display_name(x, hierarchical_categories),
    key=f"multiselect_{current_segment_id}"
)

# Display minority suggestions only if use_previous is True
if use_previous and minority_cats:
    st.caption("Minority suggestions (present in some runs but not majority): " + ", ".join(f"`{cat}`" for cat in minority_cats))

# --- Save and Next Button ---
st.markdown("""
<style>
  .button-container > button {
    width: 100% !important;
    min-height: 45px !important;
    font-size: 16px !important;
  }
  /* Optionally, center the button container */
  .button-container {
    width: 100%;
    display: flex;
    justify-content: center;
    margin-bottom: 10px; /* Add some space below the main button */
  }
  /* Ensure download button also takes full width for consistency */
  div[data-testid="stDownloadButton"] > button {
      width: 100% !important;
      min-height: 45px !important;
      font-size: 16px !important;
  }

</style>
""", unsafe_allow_html=True)

st.markdown("<div class='button-container'>", unsafe_allow_html=True)
save_button = st.button("Save and Next", key=f"save_next_button_{current_segment_id}")
st.markdown("</div>", unsafe_allow_html=True)

if save_button:
    save_classification(
        current_segment_id,
        selected_final_categories,
        st.session_state.cat_to_name_map,
        st.session_state.output_path
    )
    # Also store the classification in the session state list
    final_topic_names_for_session = sorted([st.session_state.cat_to_name_map[cat] for cat in selected_final_categories if cat in st.session_state.cat_to_name_map])
    session_record = {
        "id": current_segment_id,
        "final_topics": final_topic_names_for_session
    }
    st.session_state.session_classifications.append(session_record)

    st.session_state.current_index += 1
    # Check if the new index is a multiple of 5
    if st.session_state.current_index % 5 == 0 and st.session_state.current_index > 0:
         st.toast(f"5 more classifications saved! You can download the updated '{st.session_state.output_path.name}' file now.", icon="📥")

    st.rerun() # Ensure rerun happens cleanly
