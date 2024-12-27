from fastapi import FastAPI, Form, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse
from transformers import pipeline
import os

# Initialize FastAPI app
app = FastAPI()

# Load the TTS pipeline
tts_pipeline = pipeline("text-to-speech", model="ai4bharat/indic-parler-tts")

# Directory for generated audio files
OUTPUT_DIR = "output_audio"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class TextToSpeechRequest(BaseModel):
    text: str
    language: str  # "ml" for Malayalam, "en" for English

@app.post("/generate-tts/")
async def generate_tts(request: TextToSpeechRequest):
    """
    Generate TTS audio for Malayalam ('ml') or English ('en') text.
    """
    try:
        # Validate language
        if request.language not in ["ml", "en"]:
            raise HTTPException(
                status_code=400,
                detail="Unsupported language. Use 'ml' for Malayalam or 'en' for English."
            )

        # Generate TTS audio
        speech = tts_pipeline(request.text, lang=request.language)
        
        # Save audio to file
        output_path = os.path.join(OUTPUT_DIR, f"output_{request.language}.wav")
        with open(output_path, "wb") as f:
            f.write(speech["audio"])

        return FileResponse(output_path, media_type="audio/wav", filename=f"output_{request.language}.wav")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
