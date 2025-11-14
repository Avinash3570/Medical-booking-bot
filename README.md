# Medical Booking Bot

A Flask-based chatbot application for medical appointment booking using LangChain, Pinecone, and Groq LLM.

## Features

- **RAG-based Q&A**: Uses retrieval-augmented generation for answering medical queries
- **Smart Booking**: Intelligent appointment booking with form validation
- **Session Management**: Maintains conversation history and booking state
- **PDF Knowledge Base**: Processes medical documents for context-aware responses

## Setup Instructions

### 1. Environment Setup

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Environment Variables

1. Copy the `.env` file and update with your API keys:
```bash
# API Keys - Replace with your actual keys
PINECONE_API_KEY=your_pinecone_api_key_here
GROQ_API_KEY=your_groq_api_key_here

# Flask Configuration
FLASK_SECRET_KEY=your_secret_key_here

# Pinecone Configuration
PINECONE_INDEX_NAME=your_index_name
```

### 3. Get API Keys

#### Pinecone API Key
1. Sign up at [Pinecone](https://www.pinecone.io/)
2. Create a new project
3. Get your API key from the dashboard
4. Update `PINECONE_API_KEY` in `.env`

#### Groq API Key
1. Sign up at [Groq](https://groq.com/)
2. Navigate to API section
3. Generate a new API key
4. Update `GROQ_API_KEY` in `.env`

### 4. Initialize Vector Database

1. Place your PDF documents in the `data/` folder
2. Run the indexing script:
```bash
python store_index.py
```

### 5. Run the Application

```bash
python app.py
```

The application will be available at `http://localhost:8080`

## Project Structure

```
├── app.py                 # Main Flask application
├── store_index.py         # Vector database setup
├── src/
│   ├── helper.py         # PDF processing utilities
│   └── prompt.py         # System prompts
├── templates/
│   ├── chat.html         # Chat interface
│   └── booking_form.html # Booking form
├── static/
│   ├── style.css         # Main styles
│   └── booking_form.css  # Booking form styles
├── data/                 # PDF documents
├── .env                  # Environment variables
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## API Endpoints

- `/` - Main chat interface
- `/get` - POST endpoint for chat messages
- `/book` - Booking form with pre-filled data
- `/logout` - Clear session data
- `/session` - View current session data

## Security Notes

- Never commit `.env` file to version control
- Use strong, unique API keys
- Rotate API keys regularly
- Use HTTPS in production

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed with correct versions
2. **API Key Errors**: Verify API keys are correctly set in `.env` file
3. **Pinecone Connection**: Check if your Pinecone index exists and is accessible
4. **PDF Processing**: Ensure PDF files are in the `data/` directory

### Debug Mode

The application runs in debug mode by default. For production, set:
```python
app.run(host="0.0.0.0", port=8080, debug=False)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request


