from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
import os
import logging
import json
from pathlib import Path
from pydantic import BaseModel
from typing import Optional
from openai import AsyncOpenAI

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Question Assistant")
api_router = APIRouter(prefix="/api")


class QuestionRequest(BaseModel):
    text: str
    context: Optional[str] = None

class QuestionResponse(BaseModel):
    correct_answer: str
    explanation: str
    confidence: str
    question_type: str


@api_router.get("/")
async def root():
    return {"status": "ok", "message": "AI Question Assistant is running"}

@api_router.post("/analyze-question", response_model=QuestionResponse)
async def analyze_question(request: QuestionRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set in environment")

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )

    system_prompt = """You are an expert educational assistant specializing in physics and electrical engineering.
Analyze the given question and identify the correct answer.

Respond ONLY with valid JSON, no markdown, no extra text:
{
  "correct_answer": "The option letter (A, B, C, D) or short answer if not multiple choice",
  "explanation": "Clear, concise explanation of why this answer is correct",
  "confidence": "High or Medium or Low",
  "question_type": "multiple_choice or theoretical or calculation or not_a_question"
}

If it's not a question at all:
{
  "correct_answer": "N/A",
  "explanation": "This does not appear to be a question",
  "confidence": "N/A",
  "question_type": "not_a_question"
}"""

    try:
        response = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze this:\n\n{request.text}"}
            ],
            max_tokens=500,
            temperature=0.1,
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown code blocks if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        result = json.loads(raw)
        return QuestionResponse(
            correct_answer=result.get("correct_answer", "Unknown"),
            explanation=result.get("explanation", "No explanation available"),
            confidence=result.get("confidence", "Low"),
            question_type=result.get("question_type", "unknown")
        )

    except json.JSONDecodeError:
        logger.error(f"Failed to parse JSON response: {raw}")
        return QuestionResponse(
            correct_answer="Parse error",
            explanation=raw[:300] if raw else "Empty response",
            confidence="Low",
            question_type="unknown"
        )
    except Exception as e:
        logger.error(f"API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
