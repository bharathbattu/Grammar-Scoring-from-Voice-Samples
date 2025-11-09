"""
Pydantic schemas for API request/response models.

This module defines the data structures used in the Grammar Scoring API.
These schemas provide:
- Type validation
- Automatic JSON serialization
- OpenAPI documentation
- Clear API contracts
"""

from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field


class ASRResult(BaseModel):
    """
    Automatic Speech Recognition result containing transcription and metadata.

    Attributes:
        transcript: Full text transcription from audio
        word_count: Number of words in transcript
        duration_sec: Audio duration in seconds
        language: Detected or specified language code (e.g., 'en', 'es')
    """
    transcript: str = Field(
        ...,
        description="Full transcription text from audio file",
        example=(
            "Hello, my name is John and I am applying for this position."
        )
    )
    word_count: int = Field(
        ...,
        ge=0,
        description="Total number of words in transcript",
        example=11
    )
    duration_sec: float = Field(
        ...,
        ge=0.0,
        description="Audio duration in seconds",
        example=5.2
    )
    language: str = Field(
        default="en",
        description="Detected or specified language code",
        example="en"
    )

    class Config:
        """Pydantic configuration."""
        orm_mode = True
        schema_extra: Dict[str, Any] = {
            "example": {
                "transcript": (
                    "Hello, my name is John and I am applying for this "
                    "position."
                ),
                "word_count": 11,
                "duration_sec": 5.2,
                "language": "en"
            }
        }


class ScoreBreakdown(BaseModel):
    """
    Detailed breakdown of scoring metrics and penalties.

    Attributes:
        grammar_errors: Number of grammar/spelling errors detected
        fillers: Number of filler words detected (um, uh, like, etc.)
        wer: Word Error Rate if reference transcript provided (0.0-1.0), None otherwise
        wpm: Words per minute (speaking rate), None if duration unavailable
        normalized: Normalized penalty values [0, 1] for each component
        final_score: Final proficiency score [0, 100]
    """
    grammar_errors: int = Field(
        ...,
        ge=0,
        description="Total grammar and spelling errors detected",
        example=2
    )
    fillers: int = Field(
        ...,
        ge=0,
        description="Total filler words detected",
        example=3
    )
    wer: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "Word Error Rate (0.0-1.0) if reference provided, None otherwise"
        ),
        example=0.12
    )
    wpm: Optional[float] = Field(
        None,
        ge=0.0,
        description="Words per minute (speaking rate)",
        example=127.5
    )
    normalized: Dict[str, float] = Field(
        ...,
        description=(
            "Normalized penalty values [0, 1] for each scoring component"
        ),
        example={
            "grammar": 0.17,
            "fillers": 0.27,
            "wer": 0.34,
            "fluency": 0.12
        }
    )
    final_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Final language proficiency score [0, 100]",
        example=78.5
    )

    class Config:
        """Pydantic configuration."""
        orm_mode = True
        schema_extra: Dict[str, Any] = {
            "example": {
                "grammar_errors": 2,
                "fillers": 3,
                "wer": 0.12,
                "wpm": 127.5,
                "normalized": {
                    "grammar": 0.17,
                    "fillers": 0.27,
                    "wer": 0.34,
                    "fluency": 0.12
                },
                "final_score": 78.5
            }
        }


class GrammarError(BaseModel):
    """
    Individual grammar error detail from LanguageTool.

    Attributes:
        message: Human-readable error description
        rule_id: LanguageTool rule identifier
        context: Text context showing the error
        suggestions: Suggested corrections (up to 3)
    """
    message: str = Field(
        ...,
        description="Error description",
        example="Use 'doesn't' instead of 'dont'"
    )
    rule_id: str = Field(
        ...,
        description="LanguageTool rule identifier",
        example="EN_CONTRACTION_SPELLING"
    )
    context: str = Field(
        ...,
        description="Text context around the error",
        example="She dont like apples."
    )
    suggestions: List[str] = Field(
        default_factory=list,
        description="Suggested corrections",
        example=["doesn't"]
    )

    class Config:
        """Pydantic configuration."""
        orm_mode = True


class ScoreResponse(BaseModel):
    """
    Complete API response for audio scoring endpoint.

    This is the main response schema returned by POST /score.
    Contains all transcription, analysis, and scoring results.

    Attributes:
        asr: Transcription results and metadata
        metrics: Detailed scoring breakdown
        grammar_details: List of specific grammar errors found
        filler_words: List of detected filler words
        explanation: Human-readable score explanation
        model_version: ASR model used (e.g., 'whisper-small')
        generated_at: Timestamp when score was generated
    """
    asr: ASRResult = Field(
        ...,
        description="ASR transcription results"
    )
    metrics: ScoreBreakdown = Field(
        ...,
        description="Scoring metrics and breakdown"
    )
    grammar_details: List[GrammarError] = Field(
        default_factory=list,
        description="Detailed list of grammar errors",
        max_items=50  # Limit to prevent huge responses
    )
    filler_words: List[str] = Field(
        default_factory=list,
        description="List of detected filler words",
        example=["um", "like", "you know"]
    )
    explanation: str = Field(
        ...,
        description="Human-readable breakdown of score",
        example=(
            "Score: 78.5/100 | Grammar: -6.0 pts | Fillers: -6.8 pts | "
            "WER: -6.8 pts | Fluency: -2.4 pts"
        )
    )
    model_version: str = Field(
        default="whisper-small",
        description="ASR model version used for transcription",
        example="whisper-small"
    )
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when response was generated (UTC)",
        example="2025-11-08T14:32:10.123456"
    )

    class Config:
        """Pydantic configuration."""
        orm_mode = True
        json_encoders: Dict[type, Any] = {
            datetime: lambda v: v.isoformat()  # type: ignore
        }
        schema_extra: Dict[str, Any] = {
            "example": {
                "asr": {
                    "transcript": (
                        "Um, hello. My name is John and, you know, I am "
                        "applying for this position."
                    ),
                    "word_count": 15,
                    "duration_sec": 7.5,
                    "language": "en"
                },
                "metrics": {
                    "grammar_errors": 2,
                    "fillers": 3,
                    "wer": 0.12,
                    "wpm": 120.0,
                    "normalized": {
                        "grammar": 0.17,
                        "fillers": 0.27,
                        "wer": 0.34,
                        "fluency": 0.12
                    },
                    "final_score": 78.5
                },
                "grammar_details": [
                    {
                        "message": "Possible spelling mistake found.",
                        "rule_id": "MORFOLOGIK_RULE_EN_US",
                        "context": "My name is Jhon",
                        "suggestions": ["John", "Jon"]
                    }
                ],
                "filler_words": ["um", "you know"],
                "explanation": (
                    "Score: 78.5/100 | Grammar: -6.0 pts | "
                    "Fillers: -6.8 pts | WER: -6.8 pts | Fluency: -2.4 pts"
                ),
                "model_version": "whisper-small",
                "generated_at": "2025-11-08T14:32:10.123456"
            }
        }


class HealthResponse(BaseModel):
    """
    Health check response.

    Attributes:
        status: Service health status
        timestamp: Current server timestamp
        version: API version
    """
    status: str = Field(
        default="ok",
        description="Service health status",
        example="ok"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Current server timestamp (UTC)"
    )
    version: str = Field(
        default="1.0.0",
        description="API version",
        example="1.0.0"
    )

    class Config:
        """Pydantic configuration."""
        json_encoders: Dict[type, Any] = {
            datetime: lambda v: v.isoformat()  # type: ignore
        }
        schema_extra: Dict[str, Any] = {
            "example": {
                "status": "ok",
                "timestamp": "2025-11-08T14:32:10.123456",
                "version": "1.0.0"
            }
        }
