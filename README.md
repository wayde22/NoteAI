# NoteAI

A powerful note-taking application that combines the simplicity of Obsidian with AI capabilities. NoteAI helps you manage, search, and interact with your notes using advanced AI features while maintaining privacy.

## Features

- 📝 Markdown-based note management with Obsidian compatibility
- 🔍 Local vector search using Qdrant
- 🤖 AI-powered note querying with OpenAI integration
- 🔒 Privacy-focused with local processing
- 💬 Interactive chat interface
- 🏷️ Tag-based organization
- 🔄 Incremental and full indexing options
- 💾 Configurable chat memory
- 📊 Clean, modern UI

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
5. Run the application:
   ```bash
   streamlit run src/app.py
   ```

## Project Structure

```
NoteAI/
├── src/
│   ├── app.py              # Main Streamlit application
│   ├── indexer/            # Note indexing and vectorization
│   ├── database/           # Qdrant database management
│   ├── ai/                 # OpenAI integration
│   ├── utils/              # Utility functions
│   └── models/             # Data models
├── tests/                  # Test files
├── .env                    # Environment variables
└── requirements.txt        # Project dependencies
```

## Privacy

- All note processing and embedding generation happens locally
- OpenAI API calls are optional and can be disabled
- No data is stored in the cloud
- Vector database runs locally on your machine

## License

MIT License 