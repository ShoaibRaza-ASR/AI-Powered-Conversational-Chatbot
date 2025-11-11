# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from openai import OpenAI




from starlette.concurrency import run_in_threadpool
from typing import List, Any
import logging



import openai

logger = logging.getLogger("uvicorn.error")  # or logger = logging.getLogger(__name__)

async def call_chat_model_safe(messages: List[Any]) -> str:
    """
    Robust wrapper to call ChatOpenAI correctly across LangChain versions.
    - messages: list of langchain_core message objects [SystemMessage, HumanMessage, ...]
    Returns generated text (string).
    """
    # Try .generate(batch_of_message_lists)
    try:
        if hasattr(chat_model, "generate"):
            # generate expects a batch: list of message-lists
            batch = [messages]
            result = await run_in_threadpool(chat_model.generate, batch)
            # result.generations is typically nested list: generations[0][0].text
            try:
                return result.generations[0][0].text
            except Exception:
                # fallback try attributes
                if hasattr(result, "generations"):
                    gens = result.generations
                    if gens and gens[0] and hasattr(gens[0][0], "text"):
                        return gens[0][0].text
                # last resort
                return str(result)
    except Exception as e:
        logger.debug("generate() not available or failed: %s", e)

    # Try .predict_messages(messages) which returns an AIMessage-like object
    try:
        if hasattr(chat_model, "predict_messages"):
            result = await run_in_threadpool(chat_model.predict_messages, messages)
            # result may be AIMessage or string
            text = getattr(result, "content", None) or getattr(result, "text", None)
            if text:
                return text
            return str(result)
    except Exception as e:
        logger.debug("predict_messages() not available or failed: %s", e)

    # Try .apredict_messages (async) via run_in_threadpool fallback
    try:
        if hasattr(chat_model, "apredict_messages"):
            result = await run_in_threadpool(chat_model.apredict_messages, messages)
            text = getattr(result, "content", None) or getattr(result, "text", None)
            if text:
                return text
            return str(result)
    except Exception as e:
        logger.debug("apredict_messages() not available or failed: %s", e)

    # As a final fallback, raise an informative error
    raise RuntimeError("No supported call method found on ChatOpenAI instance. "
                       "Expected methods: generate / predict_messages / apredict_messages.")


from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found in environment. Add it to your .env file.")

client = OpenAI(api_key=OPENAI_API_KEY)
#openai.api_key = OPENAI_API_KEY

app = FastAPI()

# Initialize OpenAI Chat model using langchain_openai
# Note: model name may vary depending on your API access (gpt-4, gpt-4o, gpt-3.5-turbo, etc.)
chat_model = ChatOpenAI(model="gpt-4", temperature=0.7)

# Global context variable
conversation_context: List[str] = []

# Pydantic models for request bodies
class ChatInput(BaseModel):
    input: str

class ContextInput(BaseModel):
    context: str

class SentimentInput(BaseModel):
    input: str

class EntitiesInput(BaseModel):
    input: str

# Chat endpoint: uses conversation context + new input
@app.post("/chat")
async def chat_endpoint(chat_input: ChatInput):
    try:
        messages = []
        if conversation_context:
            context_text = "\n".join(conversation_context)
            messages.append(SystemMessage(content=f"Context:\n{context_text}"))
        messages.append(HumanMessage(content=chat_input.input))

          # Call the chat model using the safe wrapper
        bot_text = await call_chat_model_safe(messages)


        

        conversation_context.append(f"User: {chat_input.input}")
        conversation_context.append(f"Bot: {bot_text}")

        return {"response": bot_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Context endpoints
@app.get("/context")
async def get_context():
    return {"context": "\n".join(conversation_context)}

@app.post("/context")
async def set_context(context_input: ContextInput):
    global conversation_context
    conversation_context = [context_input.context]
    return {"message": "Context updated successfully"}


# Sentiment analysis endpoint (uses OpenAI completion API)
@app.post("/sentiment")
async def analyze_sentiment(sentiment_input: SentimentInput):
    try:
        prompt = f"Analyze the sentiment of this text as positive, negative or neutral:\n\"{sentiment_input.input}\""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a sentiment analysis assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            temperature=0
        )

        sentiment = response.choices[0].message.content.strip().lower()
        if sentiment not in ("positive", "negative", "neutral"):
            sentiment = "neutral"

        return {"sentiment": sentiment}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





# Entities extraction endpoint
@app.post("/entities")
async def extract_entities(entities_input: EntitiesInput):
    try:
        prompt = (
            f"Extract all key entities (people, places, organizations, etc.) from this text:\n"
            f"\"{entities_input.input}\"\nReturn as a JSON array of strings."
        )
        completion = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=150,
            temperature=0,
            stop=None
        )
        entities_text = completion.choices[0].text.strip()
        import json
        try:
            entities = json.loads(entities_text)
        except Exception:
            # fallback parsing when model returns a simple list text
            entities = [e.strip().strip('"').strip("'") for e in entities_text.strip("[]").split(",") if e.strip()]
        return {"entities": entities}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
