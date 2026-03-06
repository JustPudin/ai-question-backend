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

# 1. Configuración de rutas
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# CARGAR EL ARCHIVO DE TEORÍA AL INICIAR
def cargar_teoria():
    try:
        ruta = ROOT_DIR / "teoria.txt"
        with open(ruta, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error cargando teoria.txt: {e}")
        return "No hay contexto adicional disponible."

CONOCIMIENTO_BASE = cargar_teoria()

app = FastAPI(title="AI Physics Expert")

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

class QuestionResponse(BaseModel):
    correct_answer: str
    explanation: str
    confidence: str
    question_type: str

@api_router.post("/analyze-question", response_model=QuestionResponse)
async def analyze_question(request: QuestionRequest):
    api_key = os.environ.get("GROQ_API_KEY")
    client = AsyncOpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

    # Inyección de teoría en el System Prompt
    system_prompt = f"""You are a Ph.D. Professor in Physics.
Use the following OFFICIAL KNOWLEDGE BASE to answer:
---
{CONOCIMIENTO_BASE}
---

INSTRUCTIONS:
1. Always prioritize formulas from the context above (e.g., Transformer ratios, Lorentz Force).
2. If the question is about specific values (like Neodymium magnetism), use the tables in the context.
3. Respond ONLY in valid JSON.

JSON FORMAT:
{{
  "correct_answer": "Concise answer",
  "explanation": "Step-by-step reasoning",
  "confidence": "High",
  "question_type": "theoretical"
}}"""

    try:
        response = await client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b", # MODELO DE RAZONAMIENTO SUPERIOR
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.text}
            ],
            temperature=0.1 # Muy bajo para evitar que invente
        )

        raw = response.choices[0].message.content.strip()
        # Limpiar posibles etiquetas de razonamiento <think>
        if "</think>" in raw:
            raw = raw.split("</think>")[-1].strip()

        # Limpieza de markdown
        if "
http://googleusercontent.com/immersive_entry_chip/0
http://googleusercontent.com/immersive_entry_chip/1

### ¿Qué hacer ahora?
#1. **Crea el archivo `teoria.txt`** con el texto que te pasé arriba.
#2. **Actualiza tu `server.py`** con el nuevo código.
#3. **Reinicia tu servidor en Railway** (haz push de los cambios).

#Con esto, la IA ya no adivinará; consultará el manual de máquinas eléctricas antes de responderte. ¡Mucha suerte en el trabajo!