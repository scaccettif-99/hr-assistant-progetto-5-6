# utils.py
from config import Config
from openai import OpenAI
import asyncio

client = OpenAI(base_url=Config.AI_API_URL, api_key=Config.AI_API_KEY)


class LLMHelper:

    @staticmethod
    def chat(messages):
        """Streaming chat - rimane sincrono perché viene usato in loop"""
        return client.chat.completions.create(
            model=Config.LLM_MODEL, messages=messages, stream=True
        )

    @staticmethod
    async def get_candidate_name(context):
        """Esegui in thread separato per non bloccare Chainlit"""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model=Config.LLM_MODEL_LOW,
                messages=[
                    {
                        "role": "user",
                        "content": f"Dato il seguente contesto individua il nome e cognome del candidato e ritorna solo il nome e cognome del candidato. Quello che sto per fornirti e' il curriculum vite del candidato: {context}",
                    }
                ],
            )
        )
        return response.choices[0].message.content

    @staticmethod
    async def get_db_stats(context):
        """Esegui in thread separato per non bloccare Chainlit"""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model=Config.LLM_MODEL_LOW,
                messages=[
                    {
                        "role": "user",
                        "content": f"Il tuo compito è quello di descrivere in modo testuale, ma sintetico, le statistiche legate al database dei frammenti indicizzati da questo sistema. Ecco le informazioni necessarie per le statistiche da fornire: {context}",
                    }
                ],
            )
        )
        return response.choices[0].message.content
    
    @staticmethod
    def classify_intent(prompt):
        """Chiede all'LLM di classificare l'intenzione dell'utente."""
        response = client.chat.completions.create(
            model=Config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip().lower()

    @staticmethod
def create_prompt(context, question, candidate_name):
    """Crea il prompt per la chat"""
    return f"""Sei un esperto di recruiting specializzato nell'identificazione di candidati idonei.

Domanda dell'utente: {question}

Candidato identificato: {candidate_name}

Context dal CV:
{context}

Istruzioni:
1. Valuta se il candidato è idoneo per la posizione richiesta
2. Elenca le competenze rilevanti
3. Spiega perché è/non è adatto
4. Se non è adatto, suggerisci alternative se presenti nel database
Rispondi in italiano, in modo conciso e professionale."""