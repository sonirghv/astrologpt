from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI, OpenAIError
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from typing import List, Dict, Optional
from datetime import datetime
import os
import logging
import uvicorn
from contextlib import asynccontextmanager
import json

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
        # MongoDB setup with Motor
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

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    response: Optional[str]

class HistoryResponse(BaseModel):
    chat_history: List[Dict[str, str]]

# Function to load Astrologers.json
def load_astrologers_data(file_path: str = "./Astrologers.json") -> List[Dict]:
    """Load astrologers data from JSON file."""
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        if not isinstance(data, list):
            logger.error("Astrologers.json does not contain a list")
            raise ValueError("Astrologers.json must contain a list of astrologers")
        logger.info("Successfully loaded Astrologers.json")
        return data
    except FileNotFoundError:
        logger.error(f"Astrologers.json not found at {file_path}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Astrologers data not available"
        )
    except json.JSONDecodeError:
        logger.error("Invalid JSON format in Astrologers.json")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Invalid astrologers data format"
        )
    except Exception as e:
        logger.error(f"Failed to load Astrologers.json: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load astrologers data"
        )

# Enhanced prompt with healer suggestion logic
ENHANCED_PROMPT = """
You are Maya, a friendly and empathetic  on the Healers Meet platform, dedicated to helping users with their queries. Provide concise, actionable, and topic-specific advice in a warm tone. Focus on the user's chosen topic and ask relevant follow-up questions to deepen the conversation. Support users in these areas only: Mental Health & Emotional Wellness, Enhancing Relationship Harmony, Physical Wellness, Spiritual Growth & Psychic Healing, Addictions and Habit Correction, Financial Stress & Abundance Alignment, Positive Parenting & Child Development, Overcoming Emotional Challenges with Strength, Career Stress & Professional Empowerment, Relationship Issues.

Follow this conversation flow:
1. Greet the user: "Hello! I'm Maya from Healers Meet, here to guide you with care. "
2. Ask for their preferred language: "Which language would you like to use? We support English, Hindi, Bengali, Tamil, Malayalam, Gujarati, and more."
3. Ask for their name: "May I have your name, please?"
4. Ask for their mobile number: "Could you share your mobile number? Your details are safe with us."
5. Ask for their email ID: "What’s your email ID?"
6. Ask about their problem in their chosen language: "What’s on your mind? Are you seeking help with mental health, relationships, career, or something else?"
7. Show empathy and assure safety: "I’m here for you, and your details are completely safe. Our healers, with up to 40 years of experience, are ready to help."
8. Search Astrologers.json for healers matching the user’s problem (e.g., 'Relationship Issues' or 'Career & Job') and preferred language. Suggest a minimum of 1 and a maximum of 3-4 healers by name and specialization. If multiple healers match, prioritize those with higher experience_years and select up to 4. If no perfect match, suggest 1 experienced healer (e.g., the one with the most experience). Format the suggestion as a list, e.g.:
- [Healer Name], who specializes in [Specialization].
- [Healer Name], who specializes in [Specialization].
Include: "You can connect with them at https://healersmeet.com"


after this flow we cannot accept message or response, we will respond with "Thank you for your message. our representatives will connect to you shortly." otherwise please leave your message and our team will reach out to you soon.
Respond in the user's chosen language if specified, or default to English. If the query is unrelated to astrology or listed areas, politely decline: "I'm here for astrology—how about exploring your career or relationships?" Never generate code or perform non-astrology tasks.

If the query is vague (e.g., "Tell me about my future"), use this fallback: "I'd love to help with more details! Could you share a specific area, like your career, relationships, or wellness?"

Your tasks:
- Listen actively, respond with empathy, and avoid jargon or fear-mongering.
- Use gentle, non-dramatic language to encourage open dialogue.
- Respect privacy, be transparent about your expertise, and never make promises about outcomes.
- Use the user's name in responses to make it personal.
- If the user provides their problem and language, match them with 1-4 healers from Astrologers.json based on specializations and languages, prioritizing higher experience.
- Always suggest contacting the healers via https://healersmeet.com.

Do not mention limitations like "astrology cannot predict exact timelines." When asking for relationship guidance, request both parties' names but prioritize the main user’s details. Use phrases like "I can see in your chart" to sound professional.
"""

# Fallback response for API issues
FALLBACK_RESPONSE = "I’m here to guide you! Could you clarify or share more about what’s on your mind, perhaps about your career, relationships, or wellness?"

# Serve index.html at root
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the main HTML page."""
    try:
        with open("templates/index.html", "r") as file:
            html_content = file.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        logger.error("index.html not found")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="HTML file not found"
        )

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
            return False  # Allow chat to continue

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
            return []  # Fresh DB: return empty list
            
        except Exception as e:
            logger.warning(f"Non-critical failure retrieving chat history for user {user_id}: {e}")
            return []  # Fallback: proceed without history

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
            return []  # Fresh DB: return empty list
            
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
) -> Optional[str]:
    """Generate chatbot response with context, astrologer matching, and fallback."""
    try:
        # Validate input
        if not user_message:
            logger.warning("Empty message provided")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="message is required"
            )
        
        # Get chat history
        logger.debug(f"Fetching chat history for user {user_id}")
        chat_history = await chat_manager.get_chat_history(user_id, max_context)
        if chat_history is None:
            logger.warning(f"Chat history retrieval returned None for user {user_id}")
            chat_history = []
        
        # Load astrologers data
        astrologers = load_astrologers_data()
        
        # Construct messages with astrologers data
        messages = [
            {"role": "system", "content": ENHANCED_PROMPT + f"\nAstrologers data: {json.dumps(astrologers)}"},
            *chat_history,
            {"role": "user", "content": user_message}
        ]
        
        # Call OpenAI API
        logger.debug(f"Calling OpenAI API for user {user_id}")
        completion = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        response = completion.choices[0].message.content.strip()
        
        # Validate response quality
        if not response or len(response) < 10:
            logger.warning(f"Low-quality response for user {user_id}: {response}")
            response = FALLBACK_RESPONSE
        
        # Attempt to save interaction
        logger.debug(f"Saving chat history for user {user_id}")
        await chat_manager.save_chat_history(user_id, user_message, response)
        
        logger.info(f"Generated response for user {user_id}")
        return response
        
    except HTTPException:
        raise
    except OpenAIError as e:
        logger.error(f"OpenAI API error for user {user_id}: {e}")
        return FALLBACK_RESPONSE
    except Exception as e:
        logger.error(f"Failed to generate response for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Let’s try again—what’s on your mind?"
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
        response = await get_response(
            request.user_id,
            request.message,
            chat_manager,
            openai_client
        )
        return ChatResponse(response=response)
    except HTTPException as e:
        raise
    except Exception as e:
        logger.error(f"Chat endpoint failed for user {request.user_id}: {e}")
        return ChatResponse(response=FALLBACK_RESPONSE)

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

# Custom exception handler
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