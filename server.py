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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, '.env'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Carga del documento completo ──────────────────────────────────────────────
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

# ── Dividir documento en bloques de ~1000 caracteres ─────────────────────────
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

# ── Buscar bloques relevantes por palabras clave ──────────────────────────────
def buscar_contexto_relevante(pregunta: str, bloques: list[str], max_bloques: int = 8) -> str:
    # Limpiamos la pregunta y extraemos palabras clave (más de 4 letras)
    palabras = re.findall(r'\b\w{4,}\b', pregunta.lower())
    
    # Palabras irrelevantes a ignorar
    stopwords = {
        'para', 'como', 'este', 'esta', 'cual', 'cuál', 'qué', 'que',
        'los', 'las', 'una', 'uno', 'con', 'por', 'del', 'según',
        'segun', 'entre', 'cuando', 'donde', 'sobre', 'tiene', 'puede',
        'valor', 'formula', 'fórmula', 'calcular', 'determinar', 'obtener'
    }
    palabras_clave = [p for p in palabras if p not in stopwords]

    # Puntuamos cada bloque según cuántas palabras clave contiene
    puntuaciones = []
    for i, bloque in enumerate(bloques):
        bloque_lower = bloque.lower()
        score = sum(1 for palabra in palabras_clave if palabra in bloque_lower)
        if score > 0:
            puntuaciones.append((score, i, bloque))

    # Ordenamos por relevancia descendente
    puntuaciones.sort(key=lambda x: x[0], reverse=True)

    # Tomamos los mejores bloques
    mejores_bloques = [bloque for _, _, bloque in puntuaciones[:max_bloques]]

    if not mejores_bloques:
        # Si no encuentra nada relevante, devuelve el inicio del documento
        logger.warning("No se encontraron bloques relevantes, usando inicio del documento")
        return "\n\n".join(bloques[:max_bloques])

    contexto = "\n\n---\n\n".join(mejores_bloques)
    logger.info(f"Contexto seleccionado: {len(mejores_bloques)} bloques / {len(contexto)} caracteres (antes: {len(CONOCIMIENTO_BASE)})")
    return contexto

# Pre-dividimos el documento una sola vez al iniciar el servidor
BLOQUES_DOCUMENTO = dividir_en_bloques(CONOCIMIENTO_BASE)
logger.info(f"Documento dividido en {len(BLOQUES_DOCUMENTO)} bloques")

# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(title="Physics Expert Assistant - Claude Edition")

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
    return {
        "status": "ok",
        "model": "claude-sonnet-4-20250514",
        "knowledge_loaded": len(CONOCIMIENTO_BASE) > 100,
        "bytes_loaded": len(CONOCIMIENTO_BASE),
        "bloques_totales": len(BLOQUES_DOCUMENTO),
        "path_checked": os.path.join(BASE_DIR, "teoria.txt")
    }

@api_router.post("/analyze-question", response_model=QuestionResponse)
async def analyze_question(request: QuestionRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="El texto está vacío")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Falta ANTHROPIC_API_KEY")

    client = anthropic.AsyncAnthropic(api_key=api_key)

    # ── Aquí está la optimización: solo enviamos lo relevante ─────────────────
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

    try:
        response = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            temperature=0.1,
            system=system_prompt,
            messages=[{"role": "user", "content": request.text}]
        )

        raw_content = response.content[0].text.strip()
        logger.info(f"Respuesta Claude: {raw_content[:200]}")

        json_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
        clean_json = json_match.group(0) if json_match else raw_content
        result = json.loads(clean_json)

        return QuestionResponse(
            correct_answer=str(result.get("correct_answer", "No se pudo determinar")),
            explanation=str(result.get("explanation", "Sin detalles adicionales")),
            confidence=str(result.get("confidence", "Medium")),
            question_type=str(result.get("question_type", "theoretical"))
        )

    except json.JSONDecodeError as e:
        logger.error(f"Error JSON: {e}")
        return QuestionResponse(
            correct_answer="Error de formato",
            explanation=f"Claude no devolvió JSON válido: {str(e)}",
            confidence="Low",
            question_type="error"
        )
    except anthropic.APIStatusError as e:
        logger.error(f"Error Anthropic API: {e.status_code} - {e.message}")
        return QuestionResponse(
            correct_answer="Error de API",
            explanation=f"Error Anthropic: {e.message}",
            confidence="Low",
            question_type="error"
        )
    except Exception as e:
        logger.error(f"Error inesperado: {str(e)}")
        return QuestionResponse(
            correct_answer="Error inesperado",
            explanation=f"Problema técnico: {str(e)}",
            confidence="Low",
            question_type="error"
        )

app.include_router(api_router)
