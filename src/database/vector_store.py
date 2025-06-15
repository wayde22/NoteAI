from typing import List, Optional, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
import numpy as np
from models.note import Note
import os
from pathlib import Path
import uuid
import re
import shutil # Import shutil for directory removal

# Define a namespace UUID for generating consistent note IDs
NAMESPACE_NOTE_IDS = uuid.UUID('f83a5e8f-7f6b-4e0c-8c5a-3f5d6a8e9c1b')

# Helper function to strip common markdown
def strip_markdown(text: str) -> str:
    # Remove headers
    text = re.sub(r'#{1,6}\s*(.*)', r'\1', text)
    # Remove bold/italic markers
    text = re.sub(r'(\*\*|__)(.*?)\1', r'\2', text)
    text = re.sub(r'(\*|_)(.*?)\1', r'\2', text)
    # Remove links [text](url)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    # Remove images ![alt text](url)
    text = re.sub(r'!\[(.*?)\]\(.*?\)', r'\1', text)
    # Remove blockquotes
    text = re.sub(r'^\s*>\s*', '', text, flags=re.MULTILINE)
    # Remove list markers
    text = re.sub(r'^\s*[-*+]\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s*', '', text, flags=re.MULTILINE)
    # Remove inline code
    text = re.sub(r'`(.*?)`', r'\1', text)
    # Replace multiple newlines with single space
    text = re.sub(r'\s*\n\s*', ' ', text)
    return text.strip()

class VectorStore:
    """Manages vector storage and retrieval using Qdrant."""
    
    def __init__(self, collection_name: str = "notes", path: str = None):
        """Initialize the vector store.
        
        Args:
            collection_name: Name of the Qdrant collection
            path: Path to the directory for persistent Qdrant storage.
                 If None, uses a 'qdrant_data' folder in the user's Documents folder.
        """
        if path is None:
            # Use the user's Documents folder for persistent storage
            documents_path = Path.home() / "Documents" / "NoteAI" / "qdrant_data"
            path = str(documents_path)
            
        self._qdrant_path = Path(path)
        os.makedirs(self._qdrant_path, exist_ok=True) # Ensure the directory exists
        
        self.client = QdrantClient(path=str(self._qdrant_path))
        self.collection_name = collection_name
        self._ensure_collection()

    def _ensure_collection(self):
        """Ensure the Qdrant collection exists."""
        if not self.client.collection_exists(collection_name=self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )

    def add_note(self, note: Note, embedding: List[float]):
        """Add a note and its embedding to the vector store."""
        # Generate a stable UUID based on the file_path
        note_id = str(uuid.uuid5(NAMESPACE_NOTE_IDS, note.file_path))
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=note_id,
                    vector=embedding,
                    payload={
                        "title": note.title,
                        "content": note.content,
                        "tags": note.tags,
                        "file_path": note.file_path,
                        "created_at": note.created_at.isoformat(),
                        "updated_at": note.updated_at.isoformat(),
                    },
                )
            ],
        )

    def search(self, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant notes in the vector store."""
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit,
            query_filter=None, # Optional: add filters here
        )
        
        notes = []
        seen_file_paths = set()
        for hit in search_result:
            file_path = hit.payload.get("file_path")
            if file_path and file_path not in seen_file_paths:
                notes.append({
                    "title": hit.payload.get("title"),
                    "content": hit.payload.get("content"),
                    "file_path": file_path,
                    "score": hit.score,
                })
                seen_file_paths.add(file_path)
        return notes

    def delete_note(self, note_id: str):
        """Delete a note from the vector store.
        
        Args:
            note_id: ID of the note to delete
        """
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(
                points=[note_id]
            )
        )

    def clear_collection(self):
        """Clear all data from the collection."""
        # Delete the collection to ensure all old data is purged
        self.client.delete_collection(collection_name=self.collection_name)
        # Recreate an empty collection
        self._ensure_collection()

    def close(self):
        """Closes the Qdrant client connection if it's open."""
        if self.client:
            try:
                # Qdrant client does not have an explicit 'close' method for local mode.
                # For persistent mode, it manages the connection. 
                # However, re-assigning client to None or deleting the object 
                # can help with garbage collection and releasing file handles.
                # If issues persist, consider a more explicit shutdown mechanism if Qdrant-client offers it for local.
                del self.client 
                self.client = None
            except Exception as e:
                print(f"[ERROR] Error closing Qdrant client: {e}")

    def list_all_notes(self, preview_chars: int = 100) -> List[Dict[str, str]]:
        """List all notes with title and a short content preview."""
        count_result = self.client.count(collection_name=self.collection_name, exact=True)
        if count_result.count == 0:
            return []

        points = self.client.scroll(collection_name=self.collection_name, limit=1000)[0]
        notes = []
        for point in points:
            payload = point.payload
            content = payload.get("content", "")
            title = payload.get("title", "Untitled")
            
            # Clean "Tags:" lines from content
            lines = content.splitlines()
            filtered_lines = [line for line in lines if not re.match(r"^[Tt]ags:\s*", line.strip())]
            cleaned_content = "\n".join(filtered_lines).strip()

            # Strip markdown and replace newlines for display in preview
            preview_text = strip_markdown(cleaned_content)

            # Avoid repeating the title in the preview if the preview starts with it
            if preview_text.lower().startswith(title.lower()):
                preview_text = preview_text[len(title):].strip()
                # Remove common markdown heading characters if they are now at the start
                preview_text = re.sub(r'^\s*#+\s*', '', preview_text).strip()

            notes.append({
                "title": title,
                "preview": (preview_text[:preview_chars] + ("..." if len(preview_text) > preview_chars else ""))
            })
        return notes 