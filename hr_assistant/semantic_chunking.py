import re
import numpy as np
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from config import Config


class SemanticChunking:

    def calculate_cosine_distances(sentences):

        distances = []

        # Ciclo su tutte le frasi tranne l'ultima
        for i in range(len(sentences) - 1):

            embedding_current = sentences[i]["combined_sentence_embedding"]
            embedding_next = sentences[i + 1]["combined_sentence_embedding"]

            # Calcola la similarità coseno tra due embeddings
            similarity = cosine_similarity(
                [embedding_current],
                [embedding_next]
            )[0][0]

            # Converte la similarità in distanza
            distance = 1 - similarity

            # Salva la distanza nella lista
            distances.append(distance)

            # Salva la distanza anche nel dizionario della frase
            sentences[i]["distance_to_next"] = distance

        # L'ultima frase non ha una frase successiva da confrontare

        return distances, sentences

    def combine_sentences(sentences, buffer_size=1):

        # Ciclo su tutte le frasi
        for i in range(len(sentences)):

            # Stringa che conterrà la frase combinata
            combined_sentence = ""

            # Aggiunge le frasi precedenti in base al buffer
            for j in range(i - buffer_size, i):

                # Controlla che l'indice non sia negativo
                if j >= 0:

                    combined_sentence += sentences[j]["sentence"] + " "

            # Aggiunge la frase corrente
            combined_sentence += sentences[i]["sentence"]

            # Aggiunge le frasi successive in base al buffer
            for j in range(i + 1, i + 1 + buffer_size):

                # Controlla che l'indice sia valido
                if j < len(sentences):

                    combined_sentence += " " + sentences[j]["sentence"]

            # Salva la frase combinata nel dizionario
            sentences[i]["combined_sentence"] = combined_sentence

        return sentences

    @staticmethod
    def chunk_it(txt):

        # Divide il testo in frasi usando . ? !
        single_sentences_list = re.split(r"(?<=[.?!])\s+", txt)

        print(enumerate(single_sentences_list))

        # Crea una lista di dizionari con frase e indice
        sentences = [
            {"sentence": x, "index": i}
            for i, x in enumerate(single_sentences_list)
        ]

        print(sentences)

        # Combina ogni frase con quelle vicine
        sentences = SemanticChunking.combine_sentences(sentences)

        print(sentences)

        # Inizializza il modello embeddings OpenAI
        oaiembeds = OpenAIEmbeddings(
        openai_api_key=Config.OPENAI_KEY,
        model=Config.MODEL_NAME
)

        # Genera embeddings delle frasi combinate
        embeddings = oaiembeds.embed_documents(
            [x["combined_sentence"] for x in sentences]
        )

        # Salva ogni embedding nel relativo dizionario frase
        for i, sentence in enumerate(sentences):

            sentence["combined_sentence_embedding"] = embeddings[i]

        # Calcola le distanze semantiche tra frasi vicine
        distances, sentences = SemanticChunking.calculate_cosine_distances(
            sentences
        )

        # Definisce la soglia percentile per identificare
        # i punti di rottura semantici
        breakpoint_percentile_threshold = 95

        breakpoint_distance_threshold = np.percentile(
            distances,
            breakpoint_percentile_threshold
        )

        # Conta quante distanze superano la soglia
        num_distances_above_threshold = len(
            [
                x for x in distances
                if x > breakpoint_distance_threshold
            ]
        )

        # Recupera gli indici dove la distanza supera la soglia
        # cioè dove probabilmente cambia argomento
        indices_above_thresh = [
            i for i, x in enumerate(distances)
            if x > breakpoint_distance_threshold
        ]

        # Indice iniziale del chunk
        start_index = 0

        # Lista finale dei chunks
        chunks = []

        # Crea i chunks usando i punti di rottura
        for index in indices_above_thresh:

            # Fine chunk corrente
            end_index = index

            # Prende il gruppo di frasi
            group = sentences[start_index : end_index + 1]

            # Unisce le frasi in un testo unico
            combined_text = " ".join(
                [d["sentence"] for d in group]
            )

            # Aggiunge il chunk finale
            chunks.append(combined_text)

            # Aggiorna il punto di partenza
            start_index = index + 1

        # Gestisce l'ultimo gruppo di frasi rimasto
        if start_index < len(sentences):

            combined_text = " ".join(
                [d["sentence"] for d in sentences[start_index:]]
            )

            chunks.append(combined_text)

        return chunks