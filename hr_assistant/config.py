import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    #Chromadb
    DOCUMENTS_DIR = "resumes"
    COLLECTION_NAME = "CVs"
    PERSISTENT_DIR = "data/chromadb"
    #embeddings
    EMBEDDING_PROVIDER = "local"
    MODEL_NAME = "all-mpnet-base-v2"
    MODEL_PATH = "modelli/mio_modello"
    OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
    OPENAI_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_EMBEDDINGS_KEY = os.getenv("OPENAI_API_KEY")
    # LLM Ollama
    LLM_MODEL = "llama3.2"
    LLM_MODEL_LOW = "llama3.2"
    AI_API_URL = "http://localhost:11434/v1"
    AI_API_KEY = "ollama"

    ### openai
    # LLM_MODEL = "gpt-4o"
    # LLM_MODEL_LOW = "gpt-4o-mini"
    # AI_API_URL = "https://api.openai.com/v1/"
    # AI_API_KEY = os.getenv("OPENAI_API_KEY")