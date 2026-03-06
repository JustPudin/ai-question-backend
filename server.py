from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
import os
import logging
import json
import re
from pydantic import BaseModel
from typing import Optional
from openai import AsyncOpenAI

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, '.env'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def cargar_conocimiento_teorico():
    ruta_archivo = os.path.join(BASE_DIR, "teoria.txt")
    if not os.path.exists(ruta_archivo):
        ruta_archivo = os.path.join(os.getcwd(), "teoria.txt")
    
    try:
        if os.path.exists(ruta_archivo):
            with open(ruta_archivo, "r", encoding="utf-8") as f:
                return f.read()
    except:
        pass
    return "Error cargando base de conocimiento."

CONOCIMIENTO_BASE = cargar_conocimiento_teorico()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
api_router = APIRouter(prefix="/api")

class QuestionRequest(BaseModel):
    text: str

class QuestionResponse(BaseModel):
    correct_answer: str
    explanation: str
    confidence: str
    question_type: str

@api_router.get("/")
async def root():
    return {"status": "ok", "knowledge": len(CONOCIMIENTO_BASE) > 10}

@api_router.post("/analyze-question", response_model=QuestionResponse)
async def analyze_question(request: QuestionRequest):
    api_key = os.environ.get("GROQ_API_KEY")
    client = AsyncOpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

    system_prompt = f"""You are a Physics PhD. Use this knowledge base:
{CONOCIMIENTO_BASE}
Respond ONLY with a JSON object:
{{
  "correct_answer": "...",
  "explanation": "...",
  "confidence": "High",
  "question_type": "theoretical"
}}"""

    try:
        response = await client.chat.completions.create(
            model="llama-3.3-70b-versatile", # ESTE MODELO SIEMPRE FUNCIONA
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.text}
            ],
            temperature=0.1
        )

        raw_content = response.choices[0].message.content.strip()
        
        # Extraer JSON por si el modelo pone texto extra
        json_match = re.search(r'(\{.*\})', raw_content, re.DOTALL)
        clean_json = json_match.group(1) if json_match else raw_content
        result = json.loads(clean_json)

        return QuestionResponse(
            correct_answer=str(result.get("correct_answer", "Error")),
            explanation=str(result.get("explanation", "Error")),
            confidence=str(result.get("confidence", "Low")),
            question_type=str(result.get("question_type", "theory"))
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

app.include_router(api_router)
