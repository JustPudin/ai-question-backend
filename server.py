from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
import os
import logging
import json
import re
from pathlib import Path
from pydantic import BaseModel
from typing import Optional
from openai import AsyncOpenAI

# 1. Configuración de rutas y variables de entorno
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def cargar_conocimiento_teorico():
    ruta_archivo = ROOT_DIR / "teoria.txt"
    if not ruta_archivo.exists():
        logger.error("ARCHIVO NO ENCONTRADO: teoria.txt no existe.")
        return "No hay información teórica específica cargada."
    
    try:
        with open(ruta_archivo, "r", encoding="utf-8") as f:
            contenido = f.read()
            logger.info(f"CEREBRO CARGADO: {len(contenido)} caracteres.")
            return contenido
    except Exception as e:
        logger.error(f"Error al leer teoria.txt: {e}")
        return "Error al cargar la base de conocimiento."

CONOCIMIENTO_BASE = cargar_conocimiento_teorico()

# 2. Configuración FastAPI
app = FastAPI(title="Physics Expert Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    return {"status": "ok", "knowledge_loaded": len(CONOCIMIENTO_BASE) > 100}

@api_router.post("/analyze-question", response_model=QuestionResponse)
async def analyze_question(request: QuestionRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="El texto está vacío")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY no configurada")

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )

    system_prompt = f"""You are a Ph.D. Professor in Physics. 
Analyze using ONLY this knowledge base:
{CONOCIMIENTO_BASE}

STRICT RULES:
1. Use formulas from the base. 
2. For transformers, use 'a' ratio (N1/N2).
3. Respond ONLY with a JSON object. No prose outside the JSON.

JSON Structure:
{{
  "correct_answer": "...",
  "explanation": "...",
  "confidence": "High/Medium/Low",
  "question_type": "theoretical/calculation"
}}"""

    try:
        response = await client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.text}
            ],
            max_tokens=2000,
            temperature=0.1,
        )

        raw_content = response.choices[0].message.content.strip()
        logger.info(f"Respuesta cruda recibida de Groq")

        # --- LIMPIEZA ROBUSTA DE JSON ---
        # 1. Eliminar etiquetas <think>...</think>
        raw_content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()

        # 2. Extraer solo lo que esté entre llaves { }
        json_match = re.search(r'(\{.*\}|\[.*\])', raw_content, re.DOTALL)
        if json_match:
            clean_json = json_match.group(0)
        else:
            clean_json = raw_content

        try:
            result = json.loads(clean_json)
        except json.JSONDecodeError as je:
            logger.error(f"Error de parseo JSON: {je} - Contenido: {clean_json}")
            # Fallback si el JSON viene mal formado
            return QuestionResponse(
                correct_answer="Error de formato",
                explanation=f"El modelo no devolvió un JSON válido. Respuesta cruda: {raw_content[:200]}...",
                confidence="Low",
                question_type="error"
            )

        return QuestionResponse(
            correct_answer=str(result.get("correct_answer", "Sin respuesta")),
            explanation=str(result.get("explanation", "Sin explicación")),
            confidence=str(result.get("confidence", "Low")),
            question_type=str(result.get("question_type", "unknown"))
        )

    except Exception as e:
        logger.error(f"Error crítico en el servidor: {e}")
        raise HTTPException(status_code=500, detail=str(e))

app.include_router(api_router)