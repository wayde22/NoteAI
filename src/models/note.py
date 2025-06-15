from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field
from uuid import UUID, uuid4
import re
import hashlib
import frontmatter
from pathlib import Path

class Note(BaseModel):
    """Represents a single note with its metadata and content."""

    title: str
    content: str
    file_path: str
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow) # Use utcnow for consistency
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "title": "Meeting Notes",
                "content": "# Meeting Notes\n\n- Discussed project timeline\n- Assigned tasks",
                "tags": ["meeting", "project"],
                "file_path": "notes/meeting.md"
            }
        }

    def to_markdown(self) -> str:
        """Convert note to markdown format with frontmatter."""
        frontmatter = f"""---
title: {self.title}
tags: {', '.join(self.tags)}
created: {self.created_at.isoformat()}
updated: {self.updated_at.isoformat()}
---

"""
        return frontmatter + self.content

    @classmethod
    def from_markdown(cls, markdown_content: str, file_path: str) -> "Note":
        """Parses markdown content and extracts note data.
        Supports Obsidian-style frontmatter and basic markdown parsing.
        """
        post = frontmatter.loads(markdown_content)
        
        title = post.get("title")
        tags = post.get("tags", [])
        created_at = post.get("created")
        updated_at = post.get("updated")

        note_body_content = post.content # This is the content *after* frontmatter

        # Remove lines that start with "Tags:" or "tags:" from the content body
        # This handles cases where tags are in the body rather than frontmatter.
        lines = note_body_content.splitlines()
        filtered_lines = [line for line in lines if not re.match(r"^[Tt]ags:\s*", line.strip())]
        note_body_content = "\n".join(filtered_lines).strip()

        # Fallback for title if not found in frontmatter
        if not title:
            # Try to get from first H1 heading in the *cleaned* body content
            h1_match = re.search(r"^#\s*(.*)", note_body_content, re.MULTILINE)
            if h1_match:
                title = h1_match.group(1).strip()
            else:
                # Use filename as title if no H1 and no frontmatter title
                title = Path(file_path).stem
        
        # Ensure created_at and updated_at are datetime objects
        if created_at and not isinstance(created_at, datetime):
            try: created_at = datetime.fromisoformat(created_at) 
            except: created_at = datetime.utcnow()
        if updated_at and not isinstance(updated_at, datetime):
            try: updated_at = datetime.fromisoformat(updated_at)
            except: updated_at = datetime.utcnow()

        return cls(
            title=title,
            content=note_body_content,
            tags=tags,
            created_at=created_at or datetime.utcnow(),
            updated_at=updated_at or datetime.utcnow(),
            file_path=file_path
        ) 