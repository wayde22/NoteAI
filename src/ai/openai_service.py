from typing import List, Dict, Any, Optional
import openai
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ChatMessage:
    """Represents a message in the chat history."""
    role: str
    content: str
    timestamp: datetime = datetime.utcnow()

class OpenAIService:
    """Handles interactions with OpenAI's API."""
    
    def __init__(self, api_key: str):
        """Initialize the OpenAI service.
        
        Args:
            api_key: OpenAI API key
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.chat_history: List[ChatMessage] = []
        self.max_history = 20

    def add_to_history(self, role: str, content: str):
        """Add a message to the chat history.
        
        Args:
            role: Message role (user/assistant)
            content: Message content
        """
        self.chat_history.append(ChatMessage(role=role, content=content))
        if len(self.chat_history) > self.max_history:
            self.chat_history.pop(0)

    def clear_history(self):
        """Clear the chat history."""
        self.chat_history.clear()
        print("[AI] Chat history cleared.")

    def set_history_size(self, size: int):
        """Set the maximum size of the chat history.
        
        Args:
            size: Maximum number of messages to keep
        """
        self.max_history = max(1, min(20, size))
        while len(self.chat_history) > self.max_history:
            self.chat_history.pop(0)

    def generate_response(
        self,
        query: str,
        context: List[Dict[str, Any]],
        use_history: bool = True
    ) -> str:
        """Generate a response using OpenAI's API.
        
        Args:
            query: User's question
            context: List of relevant notes for context
            use_history: Whether to include chat history
            
        Returns:
            Generated response
        """
        messages = []
        
        # Add system message with context
        context_text = "\n\n".join([
            f"Note: {note['content']}"
            for note in context
        ])

        # Check if the user is asking to list notes
        list_keywords = ["list", "show", "what notes", "all notes", "enumerate"]
        is_list_query = any(keyword in query.lower() for keyword in list_keywords)

        if is_list_query:
            # Specialized prompt for listing notes
            note_titles = []
            for i, note in enumerate(context):
                note_titles.append(f"{i+1}. {note['title']}")
            
            messages.append({
                "role": "system",
                "content": f"""You are a helpful AI assistant. The user has asked to list notes.
                Here is a list of note titles from their context:
                {context_text}
                Please respond with a numbered list of all the note titles you found in the context. Do not add any additional information unless specifically asked.
                """
            })
        else:
            # General prompt for answering questions based on context
            messages.append({
                "role": "system",
                "content": f"""You are a helpful AI assistant that answers questions about the user's notes.
                Use the following context to answer the question. If the context does not contain relevant information to answer the question, 
                then provide a general helpful answer based on your knowledge, but do not explicitly state that the answer was not found in the provided context.

                Context:
                {context_text}"""
            })
        
        # Add chat history if enabled
        if use_history:
            for msg in self.chat_history:
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        # Add current query
        messages.append({
            "role": "user",
            "content": query
        })
        
        # Generate response
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        answer = response.choices[0].message.content
        
        # Update history
        self.add_to_history("user", query)
        self.add_to_history("assistant", answer)
        
        return answer

    def generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding for the given text.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List of floats representing the embedding
        """
        response = self.client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding 

    def generate_title(self, content: str) -> str:
        """Generates a concise title for the given content using OpenAI."""
        if not self.client:
            raise ValueError("OpenAI client not initialized. Please provide an API key.")
        
        system_message = "You are an AI assistant that extracts a concise and descriptive title from the given text. The title should be short, ideally under 10 words, and accurately reflect the main topic or purpose of the text. Do not add any extra text or formatting like quotes or bullet points, just the title."
        user_message = f"Content: {content}\n\nTitle:"
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # or gpt-4, depending on preference and cost
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=20, # Keep title short
                temperature=0.7
            )
            title = response.choices[0].message.content.strip()
            # Clean up potential leading/trailing quotes or newlines from AI response
            title = title.strip('"\n ')
            return title
        except Exception as e:
            print(f"[ERROR] Error generating title with OpenAI: {e}")
            return "Untitled Note"

    def generate_tags(self, content: str) -> List[str]:
        """Generates a list of relevant tags for the given content using OpenAI."""
        if not self.client:
            raise ValueError("OpenAI client not initialized. Please provide an API key.")
        
        system_message = "You are an AI assistant that extracts relevant, comma-separated tags from the given text. Provide only the tags, do not include any extra text or formatting. Tags should be lowercase and concise."
        user_message = f"Content: {content}\n\nTags:"
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # or gpt-4
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=50, # Sufficient for several tags
                temperature=0.5
            )
            tags_str = response.choices[0].message.content.strip()
            # Clean up potential leading/trailing quotes, newlines, or extra spaces from AI response
            tags_str = tags_str.strip('"\n ')
            # Split by comma and clean each tag
            tags = [tag.strip().lower() for tag in tags_str.split(',') if tag.strip()]
            return tags
        except Exception as e:
            print(f"[ERROR] Error generating tags with OpenAI: {e}")
            return [] 