import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from database.vector_store import VectorStore
from indexer.note_indexer import NoteIndexer
from ai.openai_service import OpenAIService
from models.note import Note
import gc # Import garbage collector module
import shutil # Import shutil for directory removal
import re # Import re for filename generation
import json # Import json for proper JSON/YAML array formatting
import pathlib # Import pathlib for Path.as_uri()

def strip_markdown(content: str) -> str:
    """Removes common Markdown elements from a string."""
    # Remove Obsidian-style tags (e.g., #tag, #tag/subtag)
    content = re.sub(r'#([a-zA-Z0-9_/-]+)', '', content)
    # Remove YAML front matter
    content = re.sub(r'^---\s*$.*?^---\s*$', '', content, flags=re.DOTALL | re.MULTILINE)
    # Remove Markdown headings
    content = re.sub(r'^#+\s*', '', content, flags=re.MULTILINE)
    # Remove bold and italics
    content = re.sub(r'(\*\*|__)(.*?)\1', r'\2', content)
    content = re.sub(r'(\*|_)(.*?)\1', r'\2', content)
    # Remove links (displaying only the text)
    content = re.sub(r'\[(.*?)\]\((.*?)\)', r'\1', content)
    # Remove images (displaying nothing)
    content = re.sub(r'!\[(.*?)\]\((.*?)\)', '', content)
    # Remove blockquotes
    content = re.sub(r'^>\s*', '', content, flags=re.MULTILINE)
    # Replace all whitespace characters (including newlines) with a single space
    content = re.sub(r'\s+', ' ', content).strip()
    return content

# Load environment variables
load_dotenv()

# --- Session State Initialization ---
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = os.getenv("OPENAI_API_KEY")

if 'vault_path' not in st.session_state:
    st.session_state.vault_path = os.getenv("OBSIDIAN_VAULT_PATH")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "ai_enabled" not in st.session_state:
    st.session_state.ai_enabled = True

if "memory_enabled" not in st.session_state:
    st.session_state.memory_enabled = True

if 'memory_size' not in st.session_state:
    st.session_state.memory_size = 10

if "last_query_relevant_notes" not in st.session_state:
    st.session_state.last_query_relevant_notes = []

# Initialize session state for note creation fields
if 'note_title_input' not in st.session_state:
    st.session_state.note_title_input = ""
if 'note_content_input' not in st.session_state:
    st.session_state.note_content_input = ""
if 'note_tags_input' not in st.session_state:
    st.session_state.note_tags_input = ""
if 'note_file_path_input' not in st.session_state:
    st.session_state.note_file_path_input = ""

# Counter for dynamically keying note input fields to force re-render
if 'note_fields_key_counter' not in st.session_state:
    st.session_state.note_fields_key_counter = 0

# Flag to trigger population of note fields from AI response
if 'trigger_populate_note_fields' not in st.session_state:
    st.session_state.trigger_populate_note_fields = False

# Initialize services and store them in session_state for persistence
# These will be created only once per session, unless explicitly reset by Full Index
if 'vector_store' not in st.session_state:
    # Only initialize if not already in session state, otherwise reuse existing
    documents_path = Path.home() / "Documents" / "NoteAI" / "qdrant_data"
    path = str(documents_path)
    try:
        # Ensure directory exists for initial creation, handled by VectorStore's __init__
        st.session_state.vector_store = VectorStore(path=path)
        print(f"[DEBUG] VectorStore ID: {id(st.session_state.vector_store)}")
    except RuntimeError as e: # Catch the specific Qdrant lock error on initial load
        if "Storage folder" in str(e) and "is already accessed" in str(e):
            st.error("Qdrant database is locked. Please restart the Streamlit app to continue.")
            st.stop() # Stop the script execution
        else:
            raise e # Re-raise other RuntimeErrors

if 'openai_service' not in st.session_state:
    st.session_state.openai_service = OpenAIService(st.session_state.openai_api_key)

if 'note_indexer' not in st.session_state:
    st.session_state.note_indexer = NoteIndexer(
        vault_path=st.session_state.vault_path,
        vector_store=st.session_state.vector_store, # Pass the cached vector_store
        openai_api_key=st.session_state.openai_api_key
    )

def main():
    # --- Get cached instances (for convenience, always re-assigned on rerun) ---
    vector_store = st.session_state.vector_store
    openai_service = st.session_state.openai_service
    indexer = st.session_state.note_indexer

    # Handle population of note fields from AI response if triggered
    if st.session_state.trigger_populate_note_fields:
        # These are the values that were captured when the button was pressed
        # They should already be in the session state due to direct assignment on button click
        # However, we explicitly re-assign them here to ensure the inputs are refreshed
        st.session_state.note_title_input = st.session_state.note_title_input # Re-assign to trigger update
        st.session_state.note_content_input = st.session_state.note_content_input
        st.session_state.note_tags_input = st.session_state.note_tags_input
        st.session_state.trigger_populate_note_fields = False # Reset the flag

    # Main content area layout
    st.title("NoteAI") # Only one main app title, at the very top
    st.markdown("<hr style='border: 2px solid #333;' />", unsafe_allow_html=True)
    
    # Display chat history
    for i, message in enumerate(st.session_state.chat_history):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Add "Add to Notes" button only for the last AI message
            if message["role"] == "assistant" and i == len(st.session_state.chat_history) - 1 and st.session_state.ai_enabled:
                if st.button("Add to Notes", key=f"add_ai_response_to_notes_{i}"):
                    st.session_state.note_title_input = f"AI Generated Note from Query: {st.session_state.chat_history[i-1]['content'] if i > 0 else ''}"
                    st.session_state.note_content_input = message["content"]
                    st.session_state.note_tags_input = "AI, generated"
                    st.session_state.trigger_populate_note_fields = True # Set flag to populate fields on next rerun
                    # Increment counter to force re-render of sidebar input fields
                    st.session_state.note_fields_key_counter += 1
                    st.rerun() # Rerun to update the input fields
            # Add a line after each Q&A session (i.e., after each assistant message)
            if message["role"] == "assistant" and i < len(st.session_state.chat_history) - 1:
                st.markdown("<hr style='border: 1px solid #555;' />", unsafe_allow_html=True)

    # Display relevant notes with scores (conditional display based on last query)
    relevant_notes_for_display = st.session_state.get("last_query_relevant_notes", [])
    relevance_threshold = st.session_state.get("relevance_threshold", 0.6) # Default if not set

    if relevant_notes_for_display:
        # Wrap the entire relevant notes section in a single expander
        with st.expander("Relevant Notes:", expanded=False): # Starts expanded, can be collapsed
            unique_notes_display = {}
            for note in relevant_notes_for_display:
                if note['file_path'] not in unique_notes_display and note['score'] >= relevance_threshold:
                    unique_notes_display[note['file_path']] = note

            sorted_unique_notes_display = sorted(unique_notes_display.values(), key=lambda n: n['score'], reverse=True)

            if not sorted_unique_notes_display:
                st.info("No notes found above the relevance threshold from the last query.")
            else:
                for i, note in enumerate(sorted_unique_notes_display):
                    # Display notes directly within the single expander
                    display_content = strip_markdown(note['content']).replace(note['title'], "", 1).strip()
                    display_content = display_content.replace("\n", " ")
                    st.markdown(f"**{i+1}. [{note['title']}]({pathlib.Path(note['file_path']).as_uri()})** (Score: {note['score']:.3f})")
                    st.markdown(f"```\n{display_content[:500]}...\n```") # Display up to 500 chars of content

    # Sidebar controls
    with st.sidebar:
        st.header("Settings")
        
        # AI toggle
        st.session_state.ai_enabled = st.toggle(
            "Enable AI",
            value=st.session_state.ai_enabled,
            help="Enable/disable OpenAI API integration"
        )
        
        # Memory controls
        st.subheader("Chat Memory")
        st.session_state.memory_enabled = st.toggle(
            "Enable Memory",
            value=st.session_state.memory_enabled,
            help="Enable/disable chat history"
        )
        
        st.session_state.memory_size = st.slider(
            "Memory Size",
            min_value=1,
            max_value=20,
            value=st.session_state.memory_size,
            help="Number of previous messages to remember"
        )
        
        if st.button("Clear Memory"):
            openai_service.clear_history()
            st.session_state.chat_history = []
            st.session_state.last_query_relevant_notes = [] # Clear relevant notes from display when memory is cleared
            st.success("Chat memory cleared!")
        
        # Indexing controls
        st.subheader("Indexing")
        if st.button("Full Index"):
            with st.spinner("Performing full index..."):
                # Path to the Qdrant data directory
                qdrant_data_path = Path.home() / "Documents" / "NoteAI" / "qdrant_data"

                # Explicitly close and remove old instances to release file locks
                if 'vector_store' in st.session_state and st.session_state.vector_store:
                    st.session_state.vector_store.close() # Attempt to close client gracefully
                    del st.session_state.vector_store # Remove reference from session state
                if 'note_indexer' in st.session_state:
                    del st.session_state.note_indexer # Remove reference from session state
                
                # Force garbage collection to help release resources
                gc.collect()

                # Attempt to remove the Qdrant data directory physically
                if qdrant_data_path.exists():
                    try:
                        shutil.rmtree(qdrant_data_path)
                        print(f"[STORE] Removed Qdrant data directory: {qdrant_data_path}")
                    except PermissionError:
                        st.error("Cannot perform full index: Qdrant database files are locked. Please restart the Streamlit app to clear the index.")
                        print(f"[ERROR] PermissionError: Could not remove Qdrant data directory. Please restart the app.")
                        return # Exit the function, do not proceed with indexing
                    except Exception as e:
                        st.error(f"An unexpected error occurred while clearing the index: {e}")
                        print(f"[ERROR] Unexpected error removing Qdrant data directory: {e}")
                        return
                else:
                    print(f"[STORE] Qdrant data directory not found at {qdrant_data_path}, skipping physical removal.")

                # Re-initialize VectorStore and NoteIndexer to ensure a clean slate
                # os.makedirs will be handled by VectorStore's __init__ for the potentially new path
                st.session_state.vector_store = VectorStore(path=str(qdrant_data_path))
                st.session_state.note_indexer = NoteIndexer(
                    vault_path=st.session_state.vault_path,
                    vector_store=st.session_state.vector_store,
                    openai_api_key=st.session_state.openai_api_key
                )
                # Get the new indexer instance (now guaranteed to be fresh)
                indexer = st.session_state.note_indexer
                indexer.full_index()
            st.success("Full indexing complete!")
            
        if st.button("Incremental Index"):
            with st.spinner("Performing incremental index..."):
                indexer.incremental_index()
            st.success("Incremental indexing complete!")
            
        if st.button("Clear Index"):
            if st.checkbox("I'm sure I want to clear the index"):
                vector_store.clear_collection()
                st.session_state.last_query_relevant_notes = [] # Clear relevant notes from display
                st.success("Index cleared!")

    # Note Management section (also moved into sidebar)
    st.sidebar.header("Note Management")

    # Relevance score slider
    relevance_threshold = st.sidebar.slider(
        "Minimum Relevance Score",
        min_value=0.0,
        max_value=1.0,
        value=0.6,  # Default threshold
        step=0.01,
        help="Only notes with a relevance score above this value will be displayed.",
        key="relevance_threshold_slider"
    )
    # Store relevance_threshold in session_state so it can be accessed outside the sidebar context
    st.session_state.relevance_threshold = relevance_threshold

    # Debug prints to check session state before input fields are rendered
    print(f"[DEBUG SIDEBAR] Current note_title_input: {st.session_state.note_title_input}")
    print(f"[DEBUG SIDEBAR] Current note_content_input (first 50 chars): {st.session_state.note_content_input[:50]}...")
    print(f"[DEBUG SIDEBAR] Current note_tags_input: {st.session_state.note_tags_input}")

    # Note fields for creation/update
    st.sidebar.subheader("Note Fields")
    note_title = st.sidebar.text_input("Title", key=f"note_title_input_{st.session_state.note_fields_key_counter}", value=st.session_state.note_title_input)

    # Reverted to original height, can be expanded down by dragging
    note_content = st.sidebar.text_area("Content", height=200, key=f"note_content_input_{st.session_state.note_fields_key_counter}", value=st.session_state.note_content_input)

    note_tags = st.sidebar.text_input("Tags (comma-separated, e.g., 'idea, project')", key=f"note_tags_input_{st.session_state.note_fields_key_counter}", value=st.session_state.note_tags_input)
    note_file_path = st.sidebar.text_input("File Path", key=f"note_file_path_input_{st.session_state.note_fields_key_counter}", value=st.session_state.note_file_path_input)

    if st.sidebar.button("Save Note", key="save_note_button"):
        if not note_content:
            st.sidebar.warning("Please provide content for the note.")
        else:
            try:
                final_title = note_title
                if not final_title and st.session_state.ai_enabled:
                    with st.spinner("Generating title with AI..."):
                        final_title = openai_service.generate_title(note_content)
                        if not final_title:  # Fallback if AI generation fails or returns empty
                            final_title = "Untitled Note"
                    st.sidebar.info(f"AI generated title: '{final_title}'")
                elif not final_title:
                    final_title = "Untitled Note"

                # Process tags
                final_tags = [tag.strip() for tag in note_tags.split(',') if tag.strip()]
                if not final_tags and st.session_state.ai_enabled:
                    with st.spinner("Generating tags with AI..."):
                        # Call OpenAI service to generate tags
                        generated_tags = openai_service.generate_tags(note_content)
                        if generated_tags:  # If AI generated tags
                            final_tags.extend(generated_tags)
                            st.sidebar.info(f"AI generated tags: {', '.join(generated_tags)}")

                # Determine the file path
                final_file_name = note_file_path.strip()
                if not final_file_name:
                    # Generate a safe filename from the title if no path is provided
                    safe_title = re.sub(r'[^a-zA-Z0-9_.-]', '_', final_title)
                    final_file_name = f"{safe_title}.md"
                    st.sidebar.info(f"Generated filename: '{final_file_name}'")

                # We'll need to create a full path for the file in the vault.
                full_file_path = Path(st.session_state.vault_path) / final_file_name

                # Construct markdown content with frontmatter for title and tags
                # Use json.dumps for the tags list to ensure proper YAML array formatting
                tags_yaml_format = json.dumps(final_tags)
                markdown_output = f"""---
title: {final_title}
tags: {tags_yaml_format}
---

{note_content}"""

                # To save the file, we'll write the content to it.
                full_file_path.write_text(markdown_output)

                # Index the new/updated file
                indexer.index_file(str(full_file_path))
                st.sidebar.success(f"Note '{final_title}' saved and indexed!")
            except Exception as e:
                st.sidebar.error(f"Error saving note: {e}")

    if st.sidebar.button("Delete Note", key="delete_note_button"):
        if not note_file_path:
            st.sidebar.warning("Please provide the file path of the note to delete.")
        else:
            try:
                full_file_path = Path(st.session_state.vault_path) / note_file_path
                if full_file_path.exists():
                    # Delete the file from the file system
                    os.remove(full_file_path)
                    # Remove the note from the index
                    indexer.remove_file(str(full_file_path))
                    st.sidebar.success(f"Note '{note_file_path}' deleted!")
                else:
                    st.sidebar.warning(f"Note at path '{note_file_path}' not found.")
            except Exception as e:
                st.sidebar.error(f"Error deleting note: {e}")

    if st.sidebar.button("Clear Note Fields", key="clear_note_fields_button"):
        st.session_state.note_title_input = ""
        st.session_state.note_content_input = ""
        st.session_state.note_tags_input = ""
        st.session_state.note_file_path_input = ""
        st.rerun()

    # Use st.chat_input for the main query bar at the bottom, and all related processing
    query = st.chat_input("Ask a question about your notes:")

    if query:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": query})

        # Get relevant notes for AI context
        # First, generate embedding for the query
        query_embedding = openai_service.generate_embedding(query)
        relevant_notes_for_ai_context = vector_store.search(query_embedding) # Pass the embedding to search

        # Store the relevant notes for display in session state
        st.session_state.last_query_relevant_notes = relevant_notes_for_ai_context

        if "list notes" in query.lower():
            all_notes = indexer.list_all_notes()
            if all_notes:
                note_list_markdown = "### Your Notes:\n\n"
                for i, note in enumerate(all_notes):
                    note_list_markdown += f"{i+1}. {note['title']}\n"
                st.session_state.chat_history.append({"role": "assistant", "content": note_list_markdown})
            else:
                st.session_state.chat_history.append({"role": "assistant", "content": "No notes are indexed yet."})            

        # Prepare context for the AI
        context_notes = ""
        if relevant_notes_for_ai_context:
            for note in relevant_notes_for_ai_context:
                context_notes += f"## {note['title']}\n\n{note['content']}\n\n"

        if st.session_state.ai_enabled:
            with st.spinner("AI is thinking..."):
                ai_response = openai_service.generate_response(
                    query,
                    relevant_notes_for_ai_context,
                    use_history=st.session_state.memory_enabled # Pass use_history from session state
                )
                st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
                
                # Add a button to add AI response to notes (directly after AI response)
                # Use a dynamic key for the button to avoid Streamlit key errors on reruns
                if st.button("Add to Notes", key=f"add_ai_response_to_notes_{len(st.session_state.chat_history) -1}"):
                    st.session_state.note_title_input = f"AI Generated Note from Query: {query[:50]}..."
                    st.session_state.note_content_input = ai_response
                    st.session_state.note_tags_input = "AI, generated"
                    st.session_state.trigger_populate_note_fields = True # Set flag to populate fields on next rerun
                    st.rerun() # Rerun to update the input fields
        else:
            st.session_state.chat_history.append({"role": "assistant", "content": "AI is currently disabled. Enable it in settings to get AI-powered responses."})
        
        st.rerun() # Rerun to display updated chat history and relevant notes

if __name__ == "__main__":
    main() 