# semantic_chunking.py

import re


class SemanticChunking:

    @staticmethod
    def chunk_it(text):

        # 1. Normalizza il testo
        text = text.replace("\n", " ")

        # 2. Divide per frasi (approssimazione semplice)
        sentences = re.split(r'(?<=[.!?]) +', text)

        chunks = []
        current_chunk = ""

        # 3. Costruzione chunk "logici"
        for sentence in sentences:

            # se il chunk è piccolo, aggiungi frase
            if len(current_chunk) + len(sentence) < 500:
                current_chunk += " " + sentence
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence

        # ultimo chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks
