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

# 1. Configuración de Rutas Robustas
# Usamos os.path para asegurar compatibilidad total con el sistema de archivos de Railway
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, '.env'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def cargar_conocimiento_teorico():
    # Buscamos el archivo exactamente en la raíz del proyecto
    ruta_archivo = os.path.join(BASE_DIR, "teoria.txt")
    
    logger.info(f"Intentando cargar conocimiento desde: {ruta_archivo}")
    
    if not os.path.exists(ruta_archivo):
        logger.error(f"¡ERROR CRÍTICO! No se encontró teoria.txt en {ruta_archivo}")
        # Intentamos una búsqueda secundaria en el directorio de trabajo actual
        ruta_archivo_alt = os.path.join(os.getcwd(), "teoria.txt")
        if not os.path.exists(ruta_archivo_alt):
            return "No hay información teórica cargada. Archivo no encontrado."
        ruta_archivo = ruta_archivo_alt

    try:
        with open(ruta_archivo, "r", encoding="utf-8") as f:
            contenido = f.read()
            logger.info(f"ÉXITO: Base de conocimiento cargada ({len(contenido)} caracteres)")
            return contenido
    except Exception as e:
        logger.error(f"Error al leer el archivo: {e}")
        return f"Error al cargar la base de conocimiento: {str(e)}"

# Inicialización del conocimiento
CONOCIMIENTO_BASE = cargar_conocimiento_teorico()

# 2. Configuración FastAPI
app = FastAPI(title="Physics Expert Assistant - Railway Edition")

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
    # Endpoint de diagnóstico mejorado
    return {
        "status": "ok", 
        "knowledge_loaded": len(CONOCIMIENTO_BASE) > 100,
        "bytes_loaded": len(CONOCIMIENTO_BASE),
        "path_checked": os.path.join(BASE_DIR, "teoria.txt")
    }

@api_router.post("/analyze-question", response_model=QuestionResponse)
async def analyze_question(request: QuestionRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="El texto está vacío")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.error("API KEY faltante en variables de entorno")
        raise HTTPException(status_code=500, detail="Configuración incompleta en Railway")

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )

    # Inyección de conocimiento con respaldo
    contexto_inyectado = CONOCIMIENTO_BASE if len(CONOCIMIENTO_BASE) > 50 else "Usa tus conocimientos generales de física (PhD level)."

    system_prompt = f"""You are a Ph.D. Professor in Physics. 
Analyze using ONLY this knowledge base:
{contexto_inyectado}

STRICT RULES:
1. Use formulas from the base. 
2. For transformers, use 'a' ratio (N1/N2).
3. Respond ONLY with a valid JSON object. No prose.

JSON Structure:
{{
  "correct_answer": "...",
  "explanation": "...",
  "confidence": "High/Medium/Low",
  "question_type": "theoretical/calculation"
}}"""

    try:
        response = await client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b-specdec",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.text}
            ],
            max_tokens=2000,
            temperature=0.1,
        )

        raw_content = response.choices[0].message.content.strip()

        # Limpieza de etiquetas de razonamiento (DeepSeek R1)
        raw_content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()

        # Extracción de bloque JSON
        json_match = re.search(r'(\{.*\})', raw_content, re.DOTALL)
        if json_match:
            clean_json = json_match.group(1)
        else:
            clean_json = raw_content

        try:
            result = json.loads(clean_json)
        except json.JSONDecodeError:
            # Si falla el parseo, intentamos limpiar posibles comas extras o caracteres raros
            logger.warning(f"Fallo en parseo inicial, intentando limpiar contenido: {clean_json[:100]}")
            raise ValueError("El modelo no generó un JSON válido")

        return QuestionResponse(
            correct_answer=str(result.get("correct_answer", "No se pudo determinar")),
            explanation=str(result.get("explanation", "Sin detalles adicionales")),
            confidence=str(result.get("confidence", "Medium")),
            question_type=str(result.get("question_type", "theory"))
        )

    except Exception as e:
        logger.error(f"Error procesando la pregunta: {str(e)}")
        # Devolvemos un error 200 con mensaje estructurado para que el Front no explote
        return QuestionResponse(
            correct_answer="Error en el análisis",
            explanation=f"Hubo un problema técnico: {str(e)}",
            confidence="Low",
            question_type="error"
        )

app.include_router(api_router)
