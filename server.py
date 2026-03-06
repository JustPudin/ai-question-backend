from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
import os
import logging
import json
import re
from pydantic import BaseModel
from typing import Optional
import anthropic  # ← CAMBIA: antes era "from openai import AsyncOpenAI"

# ── Configuración de rutas ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, '.env'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ── Carga del conocimiento teórico ───────────────────────────────────────────
def cargar_conocimiento_teorico():
    ruta_archivo = os.path.join(BASE_DIR, "teoria.txt")
    logger.info(f"Intentando cargar conocimiento desde: {ruta_archivo}")

    if not os.path.exists(ruta_archivo):
        logger.error(f"¡ERROR CRÍTICO! No se encontró teoria.txt en {ruta_archivo}")
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

CONOCIMIENTO_BASE = cargar_conocimiento_teorico()

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

# ── Modelos de datos ──────────────────────────────────────────────────────────
class QuestionRequest(BaseModel):
    text: str
    context: Optional[str] = None

class QuestionResponse(BaseModel):
    correct_answer: str
    explanation: str
    confidence: str
    question_type: str

# ── Endpoint de diagnóstico ───────────────────────────────────────────────────
@api_router.get("/")
async def root():
    return {
        "status": "ok",
        "model": "claude-sonnet-4-20250514",
        "knowledge_loaded": len(CONOCIMIENTO_BASE) > 100,
        "bytes_loaded": len(CONOCIMIENTO_BASE),
        "path_checked": os.path.join(BASE_DIR, "teoria.txt")
    }

# ── Endpoint principal ────────────────────────────────────────────────────────
@api_router.post("/analyze-question", response_model=QuestionResponse)
async def analyze_question(request: QuestionRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="El texto está vacío")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY faltante en variables de entorno")
        raise HTTPException(status_code=500, detail="Configuración incompleta: falta ANTHROPIC_API_KEY")

    # ── Cliente Anthropic ─────────────────────────────────────────────────────
    client = anthropic.AsyncAnthropic(api_key=api_key)

    # ── Contexto de conocimiento ──────────────────────────────────────────────
    contexto_inyectado = (
        CONOCIMIENTO_BASE
        if len(CONOCIMIENTO_BASE) > 50
        else "Usa tus conocimientos generales de física a nivel PhD."
    )

    # ── System prompt ─────────────────────────────────────────────────────────
    system_prompt = f"""Eres un Profesor PhD en Física e Ingeniería Mecatrónica.
Tu única fuente de verdad es la siguiente base de conocimiento técnico:

<knowledge_base>
{contexto_inyectado}
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
        # ── Llamada a Claude ──────────────────────────────────────────────────
        response = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            temperature=0.1,           # Respuestas deterministas y precisas
            system=system_prompt,      # ← En Claude, system va SEPARADO del array messages
            messages=[
                {"role": "user", "content": request.text}
            ]
        )

        # ── Extracción del contenido ──────────────────────────────────────────
        raw_content = response.content[0].text.strip()
        logger.info(f"Respuesta cruda de Claude (primeros 200 chars): {raw_content[:200]}")

        # ── Parseo del JSON ───────────────────────────────────────────────────
        # Claude normalmente devuelve JSON limpio, pero por seguridad
        # extraemos el bloque JSON si hubiera texto alrededor
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
        logger.error(f"Error al parsear JSON de Claude: {e} | Contenido: {raw_content[:300]}")
        return QuestionResponse(
            correct_answer="Error de formato",
            explanation=f"Claude no devolvió un JSON válido: {str(e)}",
            confidence="Low",
            question_type="error"
        )
    except anthropic.APIStatusError as e:
        logger.error(f"Error de API Anthropic: {e.status_code} - {e.message}")
        return QuestionResponse(
            correct_answer="Error de API",
            explanation=f"Error al conectar con Anthropic: {e.message}",
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
```

---

### PASO 6 — Verificar que el deploy funciona en Railway

Una vez que subas los cambios a tu repositorio, Railway redesplegará. Para verificar, visita:
```
https://tu-dominio.railway.app/api/
