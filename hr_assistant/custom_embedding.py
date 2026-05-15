import os

import ollama
from chromadb.api.types import EmbeddingFunction
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

from config import Config


class CustomEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.provider = Config.EMBEDDING_PROVIDER
        self.model_name = Config.MODEL_NAME
        self.model_path = Config.MODEL_PATH
        self.embedding_function = None

        if self.provider == "openai":
            self._setup_openai()
        elif self.provider == "local":
            self._setup_local_model()
        elif self.provider == "ollama":
            pass
        else:
            raise ValueError(
                f"EMBEDDING_PROVIDER '{self.provider}' non supportato"
            )

    def _setup_openai(self):
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=Config.OPENAI_EMBEDDINGS_KEY,
            model_name=self.model_name,
        )

    def _setup_local_model(self):
        if os.path.exists(self.model_path):
            self.embedding_function = SentenceTransformer(self.model_path)
        else:
            model = SentenceTransformer(self.model_name)
            os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
            model.save_pretrained(self.model_path)
            self.embedding_function = model

    def __call__(self, texts):
        if self.provider == "openai":
            return self.embedding_function(texts)
        if self.provider == "local":
            return self.embedding_function.encode(texts).tolist()
        if self.provider == "ollama":
            return [
                ollama.embeddings(model=self.model_name, prompt=text)["embedding"]
                for text in texts
            ]
        raise ValueError(f"EMBEDDING_PROVIDER '{self.provider}' non supportato")
