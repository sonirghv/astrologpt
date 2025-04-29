from fastapi import FastAPI, HTTPException, Depends, status, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from openai import OpenAI, OpenAIError
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from typing import List, Dict, Optional
from datetime import datetime
import os
import logging
import uvicorn
import json
import base64
from contextlib import asynccontextmanager
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request
from fastapi.staticfiles import StaticFiles

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# FastAPI app with lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown."""
    app.state.mongo_client = None
    app.state.openai_client = None
    app.state.collection = None
    try:
        # MongoDB setup
        mongodb_uri = os.getenv("MONGODB_URI")
        if not mongodb_uri:
            logger.error("MONGODB_URI not set")
            raise ValueError("MONGODB_URI is required")
        
        app.state.mongo_client = AsyncIOMotorClient(mongodb_uri, serverSelectionTimeoutMS=5000)
        await app.state.mongo_client.admin.command('ping')
        app.state.db = app.state.mongo_client["HealersMeet"]
        app.state.collection = app.state.db["users"]
        if app.state.collection is None:
            logger.error("Failed to initialize MongoDB collection")
            raise RuntimeError("MongoDB collection initialization failed")
        await app.state.collection.create_index("user_id")
        logger.info("MongoDB connected and indexed")
        
        # OpenAI setup
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.error("OPENAI_API_KEY not set")
            raise ValueError("OPENAI_API_KEY is required")
        
        app.state.openai_client = OpenAI(api_key=openai_api_key)
        if app.state.openai_client is None:
            logger.error("Failed to initialize OpenAI client")
            raise RuntimeError("OpenAI client initialization failed")
        logger.info("OpenAI client initialized")
        
        yield
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    finally:
        if app.state.mongo_client:
            app.state.mongo_client.close()
            logger.info("MongoDB connection closed")
        logger.info("Application shutdown")

app = FastAPI(title="Healers Meet API", lifespan=lifespan)

# Set up static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Serve frontend index.html at root
@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Pydantic models
class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    response: Optional[str]

class HistoryResponse(BaseModel):
    chat_history: List[Dict[str, str]]

class AudioTranscriptionRequest(BaseModel):
    user_id: str
    audio_data: str  # Base64 encoded audio

class AudioTranscriptionResponse(BaseModel):
    text: str

# Enhanced prompt
ENHANCED_PROMPT = """
You are Maya, a friendly and empathetic astrologer on the Healers Meet platform, dedicated to helping users with their queries. Provide concise, actionable, and topic-specific advice in a warm tone. Focus on the user's chosen topic and ask a relevant follow-up question to deepen the conversation. 

Support users in these CLIENT ISSUES areas: Mental Health & Emotional Wellness, Enhancing Relationship Harmony, Physical Wellness, Spiritual Growth & Psychic Healing, Addictions and Habit Correction, Financial Stress & Abundance Alignment, Positive Parenting & Child Development, Overcoming Emotional Challenges with Strength, Career Stress & Professional Empowerment, Relationship Issues.

Based on the user's needs, recommend relevant therapies from THE WORLD OF THERAPIES on Healers Meet:
- Crystal Healing: Aligning Energy with Nature's Gifts
- Reiki: Restoring Harmony to Mind, Body & Spirit
- Pranic Healing: Restoring Balance and Vitality
- Psychic Healing: Unlocking the Power of Your Soul
- Mind Brand Activation
- Lama Fera: Deep Healing
- Spiritual Growth for a Fulfilling Life
- NLP: Unlocking True Potential
- CBT: A Path to Clarity, Confidence & Growth
- Hypnotherapy: Transform Your Life
- Astrology: A Roadmap to Self-Discovery & Growth

Respond in the user's language if specified, or default to English. If the query is unrelated to astrology or the listed areas, politely suggest a relevant topic, e.g., "I specialize in astrology and healing—how about exploring your career or relationships?" Never generate code or perform non-healing tasks.

If the query is vague or unanswerable, use this fallback: "I'd love to help with more details! Could you share a specific area of your life you'd like guidance on?" Always end with a question to keep the conversation flowing.

Conversation flow:
1. Introduce yourself briefly with Healers Meet company name
2. Ask for the user's name
3. Ask which area they need help with (mental health, relationships, career, etc.)
4. Ask for their birth date and time
5. Ask about their specific problem or query
6. Provide specific guidance based on their query, keeping answers short and engaging
7. Based on their issue, recommend one of THE WORLD OF THERAPIES from Healers Meet that would benefit them most
8. After 3-5 exchanges, mention that Healers Meet has over 100 experienced counselors/therapists who can provide more personalized service you can visit https://healersmeet.com. 

Your tasks are:
- Listen actively to understand client needs and respond with empathy, avoiding jargon or fear-mongering
- Provide actionable insights based on astrological charts, not deterministic predictions
- Use gentle language to encourage open dialogue and invite questions
- Respect privacy and never make promises about outcomes
- Use the user's name to make the conversation more personal and emotional
- Always recommend a relevant therapy from Healers Meet's offerings based on their needs
- When appropriate, mention that Healers Meet offers both chat and call consultations with expert counselors

Questioning Techniques:

Start with closed questions (e.g., “Feeling stressed?”).
Shift to open questions (e.g., “How’s this affecting you?”).
Chunk for detail (e.g., “What triggered this?”).
Use Columbo technique (casual, then key question).
Lead ethically (e.g., “Exploring your chart could help, right?”).
Empower (e.g., “When did you feel in control?”).
Clarify vague responses (e.g., “Can you share more?”).
Keep questions clear, avoiding jargon.

communivation tips

When discussing relationships or issues involving multiple people, ask for the names of those involved but prioritize our main user's details. Always frame insights as "I can see in your chart" to maintain professionalism. 
Give answers in very short way, like if possible to finish the usual answers in 50 words also always follow the script.


"""

# Fallback response
FALLBACK_RESPONSE = "I'm here to guide you at Healers Meet! Could you share more about what's on your mind—perhaps about your relationships, career, or wellness journey?"

# Dependencies
def get_chat_manager():
    """Provide ChatHistoryManager instance."""
    if app.state.collection is None:
        logger.error("MongoDB collection not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available"
        )
    return ChatHistoryManager(app.state.collection)

def get_openai_client():
    """Provide OpenAI client."""
    if app.state.openai_client is None:
        logger.error("OpenAI client not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI service not available"
        )
    return app.state.openai_client

class ChatHistoryManager:
    """Handle MongoDB chat history operations."""
    
    def __init__(self, collection):
        if collection is None:
            logger.error("ChatHistoryManager received None collection")
            raise ValueError("MongoDB collection cannot be None")
        self.collection = collection

    async def save_chat_history(self, user_id: str, user_query: str, astro_response: str, max_history: int = 50) -> bool:
        """Save chat interaction to MongoDB."""
        try:
            timestamp = datetime.utcnow().isoformat()
            interaction = {
                "user": user_query,
                "astro_chatbot": astro_response,
                "timestamp": timestamp
            }
            
            result = await self.collection.update_one(
                {"user_id": user_id},
                {
                    "$push": {
                        "chat_history": {
                            "$each": [interaction],
                            "$slice": -max_history
                        }
                    },
                    "$set": {"last_updated": timestamp}
                },
                upsert=True
            )
            
            logger.info(f"Chat history saved for user {user_id}")
            return True
            
        except Exception as e:
            logger.warning(f"Non-critical failure saving chat history for user {user_id}: {e}")
            return False

    async def get_chat_history(self, user_id: str, limit: int = 10) -> List[Dict[str, str]]:
        """Retrieve chat history for API context."""
        try:
            user_doc = await self.collection.find_one(
                {"user_id": user_id},
                {"chat_history": {"$slice": -limit}}
            )
            
            if user_doc and "chat_history" in user_doc:
                messages = []
                for interaction in user_doc["chat_history"]:
                    messages.append({"role": "user", "content": interaction["user"]})
                    messages.append({"role": "assistant", "content": interaction["astro_chatbot"]})
                logger.info(f"Retrieved chat history for user {user_id}")
                return messages
            logger.info(f"No chat history found for user {user_id}")
            return []
            
        except Exception as e:
            logger.warning(f"Non-critical failure retrieving chat history for user {user_id}: {e}")
            return []

    async def get_raw_chat_history(self, user_id: str, limit: int = 10) -> List[Dict[str, str]]:
        """Retrieve raw chat history for API response."""
        try:
            user_doc = await self.collection.find_one(
                {"user_id": user_id},
                {"chat_history": {"$slice": -limit}}
            )
            
            if user_doc and "chat_history" in user_doc:
                logger.info(f"Retrieved raw chat history for user {user_id}")
                return user_doc["chat_history"]
            logger.info(f"No chat history found for user {user_id}")
            return []
            
        except Exception as e:
            logger.error(f"Failed to retrieve raw chat history for user {user_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve chat history"
            )

async def get_response(
    user_id: str,
    user_message: str,
    chat_manager: ChatHistoryManager = Depends(get_chat_manager),
    openai_client: OpenAI = Depends(get_openai_client),
    max_context: int = 10
) -> Dict[str, str]:
    """Generate chatbot response with context and fallback."""
    try:
        if not user_message:
            logger.warning("Empty message provided")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="message is required"
            )
        
        logger.debug(f"Fetching chat history for user {user_id}")
        chat_history = await chat_manager.get_chat_history(user_id, max_context)
        if chat_history is None:
            logger.warning(f"Chat history retrieval returned None for user {user_id}")
            chat_history = []
        
        messages = [
            {"role": "system", "content": ENHANCED_PROMPT},
            *chat_history,
            {"role": "user", "content": user_message}
        ]
        
        logger.debug(f"Calling OpenAI API for user {user_id}")
        completion = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        response = completion.choices[0].message.content.strip()
        
        if not response or len(response) < 10:
            logger.warning(f"Low-quality response for user {user_id}: {response}")
            response = FALLBACK_RESPONSE
        
        logger.debug(f"Saving chat history for user {user_id}")
        await chat_manager.save_chat_history(user_id, user_message, response)
        
        logger.info(f"Generated response for user {user_id}")
        return {"response": response}
        
    except HTTPException:
        raise
    except OpenAIError as e:
        logger.error(f"OpenAI API error for user {user_id}: {e}")
        return {"response": FALLBACK_RESPONSE}
    except Exception as e:
        logger.error(f"Failed to generate response for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Let's try again—what's on your mind?"
        )

async def transcribe_audio(
    user_id: str,
    audio_data: str,
    openai_client: OpenAI = Depends(get_openai_client)
) -> str:
    """Transcribe audio to text using Whisper API."""
    try:
        audio_bytes = base64.b64decode(audio_data)
        temp_audio_path = f"static/audio/temp_{user_id}_{datetime.now().timestamp()}.webm"
        os.makedirs(os.path.dirname(temp_audio_path), exist_ok=True)
        
        with open(temp_audio_path, "wb") as audio_file:
            audio_file.write(audio_bytes)
        
        with open(temp_audio_path, "rb") as audio_file:
            transcription = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        
        try:
            os.remove(temp_audio_path)
        except Exception as e:
            logger.warning(f"Failed to remove temp audio file: {e}")
        
        return transcription.text
    
    except Exception as e:
        logger.error(f"Failed to transcribe audio for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to transcribe audio"
        )

# API Endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    chat_manager: ChatHistoryManager = Depends(get_chat_manager),
    openai_client: OpenAI = Depends(get_openai_client)
):
    """Handle user chat requests."""
    try:
        response_data = await get_response(
            request.user_id,
            request.message,
            chat_manager,
            openai_client
        )
        return ChatResponse(**response_data)
    except HTTPException as e:
        raise
    except Exception as e:
        logger.error(f"Chat endpoint failed for user {request.user_id}: {e}")
        return ChatResponse(response=FALLBACK_RESPONSE)

@app.post("/transcribe", response_model=AudioTranscriptionResponse)
async def transcribe_endpoint(
    request: AudioTranscriptionRequest,
    openai_client: OpenAI = Depends(get_openai_client)
):
    """Transcribe audio to text."""
    try:
        text = await transcribe_audio(request.user_id, request.audio_data, openai_client)
        return AudioTranscriptionResponse(text=text)
    except HTTPException as e:
        raise
    except Exception as e:
        logger.error(f"Transcription endpoint failed for user {request.user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to transcribe audio"
        )

@app.get("/history/{user_id}", response_model=HistoryResponse)
async def get_history_endpoint(
    user_id: str,
    limit: int = 10,
    chat_manager: ChatHistoryManager = Depends(get_chat_manager)
):
    """Retrieve chat history for a user."""
    try:
        if limit < 1 or limit > 50:
            logger.warning(f"Invalid limit {limit} requested")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="limit must be between 1 and 50"
            )
        
        history = await chat_manager.get_raw_chat_history(user_id, limit)
        return HistoryResponse(chat_history=history)
        
    except HTTPException as e:
        raise
    except Exception as e:
        logger.error(f"History endpoint failed for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve chat history"
        )

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    user_id: str,
    chat_manager: ChatHistoryManager = Depends(get_chat_manager),
    openai_client: OpenAI = Depends(get_openai_client)
):
    """Handle WebSocket connections for real-time chat with speech-to-text."""
    await websocket.accept()
    
    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            
            if data.get("type") == "text":
                response_data = await get_response(
                    user_id,
                    data.get("message", ""),
                    chat_manager,
                    openai_client
                )
                
                await websocket.send_json({
                    "type": "text_response",
                    "response": response_data["response"]
                })
                
            elif data.get("type") == "audio":
                try:
                    transcribed_text = await transcribe_audio(
                        user_id,
                        data.get("audio_data", ""),
                        openai_client
                    )
                    
                    response_data = await get_response(
                        user_id,
                        transcribed_text,
                        chat_manager,
                        openai_client
                    )
                    
                    await websocket.send_json({
                        "type": "audio_response",
                        "transcribed_text": transcribed_text,
                        "response": response_data["response"]
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing audio: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "detail": "Failed to process audio"
                    })
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for user {user_id}")
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {e}")
        await websocket.close()

@app.exception_handler(Exception)
async def custom_exception_handler(request, exc):
    """Handle uncaught exceptions."""
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred. Please try again."}
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)