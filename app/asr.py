"""
Automatic Speech Recognition (ASR) module using Whisper.

This module provides audio transcription functionality using OpenAI's Whisper model
with support for faster-whisper for improved performance.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

# Global model cache to avoid reloading on each request
_whisper_model: Any = None
_model_type: Optional[str] = None


def _load_whisper_model(model_size: str = "small") -> Tuple[Any, str]:
    """
    Load Whisper model with lazy initialization.

    Attempts to use faster-whisper for better performance, falls back to
    standard whisper if not available.

    Args:
        model_size: Model size to load (tiny, base, small, medium, large)

    Returns:
        tuple: (model, model_type) where model_type is 'faster-whisper' or 'whisper'
    """
    global _whisper_model, _model_type

    if _whisper_model is not None:
        return _whisper_model, _model_type  # type: ignore

    # Try faster-whisper first for better performance
    try:
        from faster_whisper import WhisperModel  # type: ignore

        # Use CPU with INT8 for efficiency, or CUDA if available
        device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
        compute_type = "int8" if device == "cpu" else "float16"

        _whisper_model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )
        _model_type = "faster-whisper"
        print(f"✓ Loaded faster-whisper model: {model_size} on {device}")

    except ImportError:
        # Fall back to standard whisper
        import whisper  # type: ignore

        _whisper_model = whisper.load_model(model_size)  # type: ignore
        _model_type = "whisper"
        print(
            f"✓ Loaded whisper model: {model_size} (faster-whisper not available)")

    return _whisper_model, _model_type  # type: ignore


def transcribe(
    audio_path: str,
    model_size: str = "small",
    language: Optional[str] = None
) -> Dict[str, Any]:
    """
    Transcribe audio file to text using Whisper ASR.

    This function:
    1. Loads the Whisper model (cached after first call)
    2. Transcribes the audio file
    3. Calculates word count and duration
    4. Returns structured results

    The model is loaded once and reused across requests for efficiency.

    Args:
        audio_path: Path to audio file (wav, mp3, m4a, flac, ogg, etc.)
        model_size: Whisper model size - 'tiny', 'base', 'small', 'medium', 'large'
                   Default 'small' provides good balance of speed and accuracy
        language: Optional language code (e.g., 'en', 'es'). If None, auto-detected

    Returns:
        dict: Dictionary containing:
            - transcript (str): Full transcription text
            - word_count (int): Number of words in transcript
            - duration_sec (float): Audio duration in seconds
            - language (str): Detected or specified language code

    Raises:
        FileNotFoundError: If audio file doesn't exist
        ValueError: If audio file is empty or invalid
        RuntimeError: If transcription fails

    Example:
        >>> result = transcribe("sample.wav")
        >>> print(result["transcript"])
        "Hello, this is a test audio sample."
        >>> print(f"Duration: {result['duration_sec']:.2f}s, Words: {result['word_count']}")
        Duration: 3.45s, Words: 7
    """
    # Validate audio file exists
    audio_file = Path(audio_path)
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Check if file is empty
    if audio_file.stat().st_size == 0:
        raise ValueError(f"Audio file is empty: {audio_path}")

    try:
        # Load model (cached after first call)
        model: Any
        model_type: str
        model, model_type = _load_whisper_model(model_size)

        # Transcribe based on model type
        if model_type == "faster-whisper":
            segments: Any
            info: Any
            segments, info = model.transcribe(
                str(audio_path),
                language=language,
                beam_size=5,
                vad_filter=True,  # Voice Activity Detection to filter silence
                word_timestamps=False
            )

            # Collect transcript from segments
            transcript_parts: list[str] = []
            segment: Any
            for segment in segments:
                transcript_parts.append(str(segment.text.strip()))

            transcript: str = " ".join(transcript_parts)
            duration_sec: float = float(info.duration)
            detected_language: str = str(info.language)

        else:
            # Standard whisper
            result: Any = model.transcribe(
                str(audio_path),
                language=language
            )

            transcript = str(result["text"].strip())
            duration_sec = float(result.get("duration", 0.0))
            detected_language = str(result.get("language", language or "en"))

        # Handle empty transcription
        if not transcript:
            return {
                "transcript": "",
                "word_count": 0,
                "duration_sec": duration_sec,
                "language": detected_language
            }

        # Calculate word count (split on whitespace)
        word_count = len(transcript.split())

        return {
            "transcript": transcript,
            "word_count": word_count,
            "duration_sec": round(duration_sec, 2),
            "language": detected_language
        }

    except Exception as e:
        raise RuntimeError(f"Transcription failed for {audio_path}: {str(e)}")


def get_model_info() -> Dict[str, Optional[str]]:
    """
    Get information about the currently loaded Whisper model.

    Returns:
        dict: Model type and status information
    """
    if _whisper_model is None:
        return {
            "status": "not_loaded",
            "model_type": None,
            "message": "Model will be loaded on first transcription request"
        }

    return {
        "status": "loaded",
        "model_type": _model_type,
        "message": f"Using {_model_type} for transcription"
    }
