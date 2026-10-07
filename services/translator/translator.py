"""Translation microservice called by app/indexer.py via TRANSLATOR_URL.

Run: uvicorn translator:app --host 127.0.0.1 --port 8080
If it is not running, indexer.py falls back to Gemini for translation.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from deep_translator import GoogleTranslator


app = FastAPI(title="Translator Service", version="1.0.0")

SUPPORTED_LANGUAGES = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "Italian": "it",
}

class TranslateRequest(BaseModel):
    text: str
    source_language: str 
    target_language: str  


class TranslateResponse(BaseModel):
    translated_text: str
    source_language: str
    target_language: str


@app.get("/ok")
def ok():
    return {"status": "ok"}


@app.get("/supported-languages")
def get_supported_languages():
    return {"supported_languages": list(SUPPORTED_LANGUAGES.keys())}


@app.post("/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest):
    source_lang = req.source_language.strip().title()
    target_lang = req.target_language.strip().title()

    if source_lang == target_lang:
        return TranslateResponse(
            translated_text=req.text,
            source_language=source_lang,
            target_language=target_lang
        )

    if source_lang not in SUPPORTED_LANGUAGES or target_lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Supported languages: {', '.join(SUPPORTED_LANGUAGES)}",
        )
    source_code = SUPPORTED_LANGUAGES[source_lang]
    target_code = SUPPORTED_LANGUAGES[target_lang]

    translator = GoogleTranslator(source=source_code, target=target_code)
    translated = translator.translate(req.text)

    return TranslateResponse(
        translated_text=translated,
        source_language=source_lang,
        target_language=target_lang
    )