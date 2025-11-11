"""
FastAPI application for Grammar Scoring Engine from Voice Samples.

This module provides REST API endpoints for:
- Health check
- Audio scoring (transcription + grammar + fluency analysis)
"""

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List, cast

from fastapi import FastAPI, UploadFile, File, Form, HTTPException

# Import application modules
from app.asr import transcribe, get_model_info
from app.text_features import (
    grammar_errors,
    filler_count,
    words_per_minute,
    normalize_transcript
)
from app.scoring import (
    normalize_grammar_errors,
    normalize_fillers,
    normalize_wer,
    fluency_penalty,
    calculate_final_score,
    generate_score_explanation
)
from app.schemas import ScoreResponse, ASRResult, ScoreBreakdown, GrammarError

app = FastAPI(
    title="Grammar Scoring Engine",
    description="Automated grammar and fluency scoring from audio samples using Whisper ASR",
    version="1.0.0"
)


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint to verify API is running.

    Returns:
        dict: Status message indicating service health
    """
    return {"status": "ok"}


@app.post("/score", response_model=ScoreResponse)
async def score_audio(
    audio: UploadFile = File(...,
                             description="Audio file to transcribe and score"),
    reference_transcript: Optional[str] = Form(
        None, description="Optional reference transcript for WER calculation")
) -> ScoreResponse:
    """
    Score an audio file for grammar, fluency, and language proficiency.

    This endpoint:
    1. Saves the uploaded audio temporarily
    2. Transcribes it using Whisper ASR
    3. Extracts grammar errors, filler words, and fluency metrics
    4. Calculates WER if reference transcript is provided
    5. Computes a weighted composite score (0-100)
    6. Returns structured JSON with detailed breakdown

    Args:
        audio: Uploaded audio file (supported formats: wav, mp3, m4a, flac, ogg)
        reference_transcript: Optional ground truth transcript for accuracy calculation

    Returns:
        ScoreResponse: Detailed scoring results including transcript, errors, and final score

    Raises:
        HTTPException: If file processing or scoring fails
    """
    # Validate file type
    allowed_extensions = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}
    file_ext = Path(audio.filename).suffix.lower() if audio.filename else ""

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file format. "
                f"Allowed: {', '.join(allowed_extensions)}"
            )
        )

    tmp_audio_path = None

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await audio.read()
            tmp_file.write(content)
            tmp_audio_path = tmp_file.name

        # Transcribe audio using Whisper ASR
        asr_result = transcribe(tmp_audio_path, model_size="small")
        transcript_text = asr_result["transcript"]
        word_count = asr_result["word_count"]
        duration_sec = asr_result["duration_sec"]
        language = asr_result.get("language", "en")

        # Clean and normalize transcript
        normalized_text = normalize_transcript(transcript_text)

        # Extract grammar errors
        grammar_error_count, grammar_error_details = grammar_errors(
            normalized_text, language="en-US")

        # Detect filler words
        filler_word_count, filler_word_list = filler_count(normalized_text)

        # Calculate fluency (WPM)
        wpm = words_per_minute(word_count, duration_sec)

        # Calculate WER if reference provided
        wer_value = None
        if reference_transcript and reference_transcript.strip():
            try:
                from jiwer import wer as calculate_wer
                wer_value = calculate_wer(
                    reference_transcript, normalized_text)
            except ImportError:
                
                wer_value = None
            except Exception:
               
                wer_value = None

        # Normalize penalties
        grammar_penalty = normalize_grammar_errors(
            grammar_error_count, word_count)
        filler_penalty = normalize_fillers(filler_word_count, word_count)
        wer_penalty = normalize_wer(wer_value)
        fluency_pen = fluency_penalty(wpm)

        # Compute final score
        final = calculate_final_score(
            grammar_penalty=grammar_penalty,
            filler_penalty=filler_penalty,
            wer_penalty=wer_penalty,
            fluency_pen=fluency_pen
        )

        # Generate explanation
        explanation = generate_score_explanation(
            grammar_penalty=grammar_penalty,
            filler_penalty=filler_penalty,
            wer_penalty=wer_penalty,
            fluency_pen=fluency_pen,
            final=final
        )

        # Build response objects
        asr_obj = ASRResult(
            transcript=transcript_text,
            word_count=word_count,
            duration_sec=duration_sec,
            language=language
        )

        # Convert grammar errors to schema format (limit to top 20 for response size)
        grammar_error_objects = [
            GrammarError(
                message=err["message"],
                rule_id=err["rule_id"],
                context=err["context"],
                suggestions=cast(List[str], err.get("replacements", []))
            )
            for err in grammar_error_details[:20]
        ]

        metrics_obj = ScoreBreakdown(
            grammar_errors=grammar_error_count,
            fillers=filler_word_count,
            wer=round(wer_value, 4) if wer_value is not None else None,
            wpm=wpm,
            normalized={
                "grammar": round(grammar_penalty, 4),
                "fillers": round(filler_penalty, 4),
                "wer": round(wer_penalty, 4),
                "fluency": round(fluency_pen, 4)
            },
            final_score=final
        )

        # Get model version info
        model_info = get_model_info()
        model_version = f"{model_info.get('model_type', 'whisper')}-small"

        response = ScoreResponse(
            asr=asr_obj,
            metrics=metrics_obj,
            grammar_details=grammar_error_objects,
            filler_words=filler_word_list[:20], 
            explanation=explanation,
            model_version=model_version,
            generated_at=datetime.now(timezone.utc)
        )

        return response

    except HTTPException:
        # Re-raise HTTP exceptions
        raise

    except Exception as e:
        # Log error and return 500
        raise HTTPException(
            status_code=500,
            detail=f"Error processing audio file: {str(e)}"
        )

    finally:
       
        if tmp_audio_path and Path(tmp_audio_path).exists():
            try:
                Path(tmp_audio_path).unlink()
            except Exception:
                pass  


@app.get("/")
async def root() -> Dict[str, Any]:
    """
    Root endpoint with API information.

    Returns:
        dict: Welcome message and available endpoints
    """
    return {
        "message": "Grammar Scoring Engine API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "score": "/score (POST)",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

