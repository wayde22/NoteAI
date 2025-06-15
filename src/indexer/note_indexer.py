import os
from pathlib import Path
from typing import Set, Optional, List
# Remove watchdog imports temporarily
# from watchdog.observers import Observer
# from watchdog.events import FileSystemEventHandler
import openai
from models.note import Note
from database.vector_store import VectorStore
from database.vector_store import NAMESPACE_NOTE_IDS
import uuid

SUPPORTED_EXTENSIONS = [
    ".md", ".txt", ".text", ".pdf", ".csv", ".doc", ".docx", ".xlsx",
    ".html", ".htm", ".js", ".ts", ".kt", ".java", ".py", ".json", ".xml"
]

EXCLUDE_DIRS = {".obsidian", "qdrant_data"}

def extract_content(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    if ext in [".md", ".txt", ".text", ".js", ".ts", ".kt", ".java", ".py", ".json", ".xml"]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    elif ext == ".pdf":
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    elif ext in [".doc", ".docx"]:
        from docx import Document
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)
    elif ext == ".csv":
        import pandas as pd
        return pd.read_csv(file_path).to_string()
    elif ext in [".xlsx"]:
        import pandas as pd
        return pd.read_excel(file_path).to_string()
    elif ext in [".html", ".htm"]:
        from bs4 import BeautifulSoup
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f, "html.parser")
            return soup.get_text()
    else:
        # Fallback: try reading as text
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            return ""

# Removed FileSystemEventHandler inheritance temporarily
class NoteIndexer:
    """Handles note indexing and file system watching."""
    
    def __init__(
        self,
        vault_path: str,
        vector_store: VectorStore,
        openai_api_key: Optional[str] = None
    ):
        """Initialize the indexer.
        
        Args:
            vault_path: Path to the Obsidian vault
            vector_store: VectorStore instance for storing embeddings
            openai_api_key: OpenAI API key for generating embeddings
        """
        print(f"[DEBUG] NoteIndexer init - vector_store ID: {id(vector_store)}")
        self.vault_path = Path(vault_path)
        self.vector_store = vector_store
        # self.observer = Observer() # Removed temporarily
        self.indexed_files: Set[str] = set()
        
        if openai_api_key:
            self.client = openai.OpenAI(api_key=openai_api_key)
        else:
            self.client = None

    # Removed watchdog methods temporarily
    # def start_watching(self):
    #     """Start watching the vault directory for changes."""
    #     self.observer.schedule(self, str(self.vault_path), recursive=True)
    #     self.observer.start()

    # def stop_watching(self):
    #     """Stop watching the vault directory."""
    #     self.observer.stop()
    #     self.observer.join()

    # def on_created(self, event):
    #     """Handle file creation events."""
    #     if not event.is_directory and event.src_path.endswith('.md'):
    #         self.index_file(event.src_path)

    # def on_modified(self, event):
    #     """Handle file modification events."""-
    #     if not event.is_directory and event.src_path.endswith('.md'):
    #         self.index_file(event.src_path)

    # def on_deleted(self, event):
    #     """Handle file deletion events."""
    #     if not event.is_directory and event.src_path.endswith('.md'):
    #         self.remove_file(event.src_path)

    def index_file(self, file_path: str):
        """Index a single supported file."""
        file_path_obj = Path(file_path)

        # Check if the file path is within any excluded directory
        for excluded_dir in EXCLUDE_DIRS:
            if excluded_dir in file_path_obj.parts:
                print(f"[SKIP] Excluded directory '{excluded_dir}' detected in path: {file_path}")
                return

        ext = file_path_obj.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            print(f"[SKIP] Unsupported extension: {file_path}")
            return
        print(f"[INDEX] Attempting to index: {file_path}")
        try:
            content = extract_content(file_path)
            note = Note.from_markdown(content, file_path)
            embedding = self._generate_embedding(note.content)
            self.vector_store.add_note(note, embedding)
            self.indexed_files.add(file_path)
        except Exception as e:
            print(f"[ERROR] Error indexing {file_path}: {e}")

    def remove_file(self, file_path: str):
        """Remove a file from the index."""
        if file_path in self.indexed_files:
            # Generate the same deterministic UUID to delete from Qdrant
            note_id_to_delete = str(uuid.uuid5(NAMESPACE_NOTE_IDS, file_path))
            try:
                self.vector_store.delete_note(note_id_to_delete)
                print(f"[STORE] Note deleted from vector store: {file_path}")
            except Exception as e:
                print(f"[ERROR] Error deleting note from vector store {file_path}: {e}")
            
            self.indexed_files.remove(file_path)
        else:
            print(f"[WARN] Attempted to remove non-indexed file: {file_path}")

    def list_all_notes(self) -> List[Note]:
        """Retrieve all notes currently stored in the vector store."""
        return self.vector_store.list_all_notes()

    def full_index(self):
        """Perform a full reindex of all supported files, excluding internal directories."""
        print(f"[INDEX] Starting full index...")
        print(f"[DEBUG] Using vault path: {self.vault_path}")
        if not self.vault_path.exists():
            print(f"[ERROR] Vault path does not exist: {self.vault_path}")
        else:
            print(f"[DEBUG] Contents of vault directory:")
            for item in self.vault_path.iterdir():
                print(f"  - {item} (dir: {item.is_dir()}, file: {item.is_file()})")
        
        # No longer calling reset_and_reinitialize here, it's handled in app.py
        # Force a complete reset of the Qdrant database by deleting the data directory
        # self.vector_store.reset_and_reinitialize()
        self.indexed_files.clear()
        
        for root_str, dirs, files in os.walk(self.vault_path, topdown=True):
            root = Path(root_str)
            
            # Filter out excluded directories *before* walking into them
            dirs[:] = [d for d in dirs if (root / d).name not in EXCLUDE_DIRS]
            
            for file in files:
                file_path = str(root / file)
                
                ext = Path(file_path).suffix.lower()
                if ext in SUPPORTED_EXTENSIONS:
                    print(f"[INDEX] Found supported file: {file_path}")
                    self.index_file(file_path)
        print("[INDEX] Full index complete.")

    def incremental_index(self):
        """Perform an incremental index of new or modified supported files, excluding internal directories."""
        for root_str, dirs, files in os.walk(self.vault_path, topdown=True):
            root = Path(root_str)
            # Filter out excluded directories *before* walking into them
            dirs[:] = [d for d in dirs if (root / d).name not in EXCLUDE_DIRS]

            for file in files:
                file_path = str(root / file)
                
                ext = Path(file_path).suffix.lower()
                if ext in SUPPORTED_EXTENSIONS:
                    if file_path not in self.indexed_files:
                        self.index_file(file_path)

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding for the given text."""
        if not self.client:
            raise ValueError("OpenAI client not initialized. Please provide an API key.")
        response = self.client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding 