# Travel Assistant Chat App

A Streamlit-based chat application that provides travel assistance using OpenAI's GPT model and RAG (Retrieval-Augmented Generation) technology.

## Features

- Interactive chat interface
- RAG-powered responses using travel documents
- Real-time streaming responses
- Chat history management
- Document-based knowledge retrieval
- Web content integration

## Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/travel-assistant-chat.git
cd travel-assistant-chat
```

2. Create and activate a virtual environment:
```bash
python -m venv rag_env
# On Windows:
rag_env\Scripts\activate
# On Unix or MacOS:
source rag_env/bin/activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Running the App

To run the app, use the following command:
```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`.

## Project Structure

- `app.py`: Main Streamlit application file
- `rag_utils.py`: RAG implementation and utility functions
- `requirements.txt`: Project dependencies
- `.env`: Environment variables (not tracked in git)
- `docs/`: Directory containing travel-related documents

## Technologies Used

- Streamlit
- LangChain
- OpenAI GPT-4
- ChromaDB
- Python 3.10+

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for providing the GPT models
- LangChain for the RAG implementation framework
- Streamlit for the web application framework 