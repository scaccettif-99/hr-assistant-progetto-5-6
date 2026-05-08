import os
import chainlit as cl
from documentprocessor import DocumentProcessor
from database import Database
from confing import Config
from utils import LLMHelper

db = Database()

#process-documents

added, updated, removed = DocumentProcessor.process_documents(db)
print(f"Document sync completed: {added} added, {updated} updated, {removed} removed")

@cl.action_callback(db_stats)
async def on_action(action: cl.Action):
    print(action.payload)
    db_info = db.get_stats()
    response = await LLMHelper.get_db_stats(db_info)
    await cl.Message(response).send()