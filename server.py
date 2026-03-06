from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
import os
import logging
import json
import re
from pydantic import BaseModel
from typing import Optional
import anthropic
from openai import AsyncOpenAI

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, '.env'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Carga del documento ───────────────────────────────────────────────────────
def cargar_conocimiento_teorico():
    ruta_archivo = os.path.join(BASE_DIR, "teoria.txt")
    logger.info(f"Intentando cargar conocimiento desde: {ruta_archivo}")
    if not os.path.exists(ruta_archivo):
        ruta_archivo_alt = os.path.join(os.getcwd(), "teoria.txt")
        if not os.path.exists(ruta_archivo_alt):
            return "No hay información teórica cargada."
        ruta_archivo = ruta_archivo_alt
    try:
        with open(ruta_archivo, "r", encoding="utf-8") as f:
            contenido = f.read()
            logger.info(f"ÉXITO: Base de conocimiento cargada ({len(contenido)} caracteres)")
            return contenido
    except Exception as e:
        logger.error(f"Error al leer el archivo: {e}")
        return f"Error: {str(e)}"

CONOCIMIENTO_BASE = cargar_conocimiento_teorico()

# ── Dividir en bloques ────────────────────────────────────────────────────────
def dividir_en_bloques(texto: str, tamanio: int = 1000) -> list[str]:
    parrafos = texto.split('\n\n')
    bloques = []
    bloque_actual = ""
    for parrafo in parrafos:
        if len(bloque_actual) + len(parrafo) < tamanio:
            bloque_actual += parrafo + "\n\n"
        else:
            if bloque_actual.strip():
                bloques.append(bloque_actual.strip())
            bloque_actual = parrafo + "\n\n"
    if bloque_actual.strip():
        bloques.append(bloque_actual.strip())
    return bloques

# ── Buscar contexto relevante ─────────────────────────────────────────────────
def buscar_contexto_relevante(pregunta: str, bloques: list[str], max_bloques: int = 8) -> str:
    palabras = re.findall(r'\b\w{4,}\b', pregunta.lower())
    stopwords = {
        'para', 'como', 'este', 'esta', 'cual', 'cuál', 'qué', 'que',
        'los', 'las', 'una', 'uno', 'con', 'por', 'del', 'según',
        'segun', 'entre', 'cuando', 'donde', 'sobre', 'tiene', 'puede',
        'valor', 'formula', 'fórmula', 'calcular', 'determinar', 'obtener'
    }
    palabras_clave = [p for p in palabras if p not in stopwords]
    puntuaciones = []
    for i, bloque in enumerate(bloques):
        bloque_lower = bloque.lower()
        score = sum(1 for palabra in palabras_clave if palabra in bloque_lower)
        if score > 0:
            puntuaciones.append((score, i, bloque))
    puntuaciones.sort(key=lambda x: x[0], reverse=True)
    mejores_bloques = [bloque for _, _, bloque in puntuaciones[:max_bloques]]
    if not mejores_bloques:
        logger.warning("No se encontraron bloques relevantes, usando inicio del documento")
        return "\n\n".join(bloques[:max_bloques])
    contexto = "\n\n---\n\n".join(mejores_bloques)
    logger.info(f"Contexto: {len(mejores_bloques)} bloques / {len(contexto)} caracteres")
    return contexto

BLOQUES_DOCUMENTO = dividir_en_bloques(CONOCIMIENTO_BASE)
logger.info(f"Documento dividido en {len(BLOQUES_DOCUMENTO)} bloques")

# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(title="Physics Expert Assistant - Claude + Groq Fallback")

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
    model_used: str  # ← NUEVO: indica qué modelo respondió

# ── Llamada a Claude ──────────────────────────────────────────────────────────
async def llamar_claude(system_prompt: str, pregunta: str) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY no configurada")

    client = anthropic.AsyncAnthropic(api_key=api_key)
    response = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        temperature=0.1,
        system=system_prompt,
        messages=[{"role": "user", "content": pregunta}]
    )
    return response.content[0].text.strip()

# ── Llamada a Groq (fallback) ─────────────────────────────────────────────────
async def llamar_groq(system_prompt: str, pregunta: str) -> str:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY no configurada")

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )
    response = await client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # Mejor modelo gratuito de Groq actualmente
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": pregunta}
        ],
        max_tokens=2000,
        temperature=0.1,
    )
    raw = response.choices[0].message.content.strip()
    # Limpieza de etiquetas de razonamiento por si acaso
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    return raw

# ── Parsear JSON de respuesta ─────────────────────────────────────────────────
def parsear_respuesta(raw_content: str) -> dict:
    json_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
    clean_json = json_match.group(0) if json_match else raw_content
    return json.loads(clean_json)

# ── Endpoints ─────────────────────────────────────────────────────────────────
@api_router.get("/")
async def root():
    return {
        "status": "ok",
        "primary_model": "claude-sonnet-4-20250514",
        "fallback_model": "groq/llama-3.3-70b-versatile",
        "knowledge_loaded": len(CONOCIMIENTO_BASE) > 100,
        "bytes_loaded": len(CONOCIMIENTO_BASE),
        "bloques_totales": len(BLOQUES_DOCUMENTO),
        "path_checked": os.path.join(BASE_DIR, "teoria.txt")
    }

@api_router.post("/analyze-question", response_model=QuestionResponse)
async def analyze_question(request: QuestionRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="El texto está vacío")

    contexto_relevante = buscar_contexto_relevante(request.text, BLOQUES_DOCUMENTO)

    system_prompt = f"""Eres un Profesor PhD en Física e Ingeniería Mecatrónica.
Tu única fuente de verdad es la siguiente base de conocimiento técnico:

<knowledge_base>
{contexto_relevante}
</knowledge_base>

REGLAS ESTRICTAS:
1. Responde ÚNICAMENTE basándote en la base de conocimiento anterior.
2. Usa las fórmulas y definiciones exactas del documento.
3. Para transformadores, usa siempre la relación de transformación 'a' = N1/N2.
4. Tu respuesta debe ser EXCLUSIVAMENTE un objeto JSON válido, sin texto adicional.

ESTRUCTURA JSON OBLIGATORIA:
{{
  "correct_answer": "La respuesta correcta aquí",
  "explanation": "Explicación detallada usando fórmulas del documento",
  "confidence": "High/Medium/Low",
  "question_type": "theoretical/calculation"
}}"""

    model_used = "claude-sonnet-4-20250514"
    raw_content = None

    # ── Intento 1: Claude ─────────────────────────────────────────────────────
    try:
        logger.info("Intentando con Claude...")
        raw_content = await llamar_claude(system_prompt, request.text)
        logger.info("Claude respondió correctamente")

    except anthropic.APIStatusError as e:
        # Errores específicos que activan el fallback a Groq
        errores_de_credito = [
            "credit_balance_too_low",
            "billing",
            "quota",
            "insufficient"
        ]
        es_error_credito = any(err in str(e).lower() for err in errores_de_credito)

        if es_error_credito or e.status_code in [402, 429]:
            logger.warning(f"⚠️ Claude sin créditos ({e.status_code}). Activando fallback a Groq...")
            model_used = "groq/llama-3.3-70b-versatile"
            try:
                raw_content = await llamar_groq(system_prompt, request.text)
                logger.info("Groq respondió correctamente como fallback")
            except Exception as groq_error:
                logger.error(f"Groq también falló: {groq_error}")
                return QuestionResponse(
                    correct_answer="Sin servicio disponible",
                    explanation="Claude sin créditos y Groq no disponible. Por favor intenta más tarde.",
                    confidence="Low",
                    question_type="error",
                    model_used="none"
                )
        else:
            logger.error(f"Error de Claude no relacionado con créditos: {e}")
            return QuestionResponse(
                correct_answer="Error de API",
                explanation=f"Error en Anthropic: {e.message}",
                confidence="Low",
                question_type="error",
                model_used="claude-sonnet-4-20250514"
            )

    except Exception as e:
        logger.error(f"Error inesperado llamando Claude: {e}")
        return QuestionResponse(
            correct_answer="Error inesperado",
            explanation=f"Problema técnico: {str(e)}",
            confidence="Low",
            question_type="error",
            model_used="unknown"
        )

    # ── Parsear respuesta ─────────────────────────────────────────────────────
    try:
        result = parsear_respuesta(raw_content)
        return QuestionResponse(
            correct_answer=str(result.get("correct_answer", "No se pudo determinar")),
            explanation=str(result.get("explanation", "Sin detalles adicionales")),
            confidence=str(result.get("confidence", "Medium")),
            question_type=str(result.get("question_type", "theoretical")),
            model_used=model_used
        )
    except json.JSONDecodeError as e:
        logger.error(f"Error JSON: {e} | Contenido: {raw_content[:300]}")
        return QuestionResponse(
            correct_answer="Error de formato",
            explanation=f"No se pudo parsear la respuesta: {str(e)}",
            confidence="Low",
            question_type="error",
            model_used=model_used
        )

app.include_router(api_router)
