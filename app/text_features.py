"""
Text feature extraction module for grammar and fluency analysis.

This module provides core NLP feature extraction functions used in the scoring pipeline:
- Grammar error detection using LanguageTool
- Filler word counting
- Fluency metrics (WPM)
- Basic sentence statistics

NOTE: This module focuses purely on feature extraction. Scoring logic and weights
are handled separately in scoring.py. These functions are designed to be pure,
testable, and reusable across the FastAPI pipeline.
"""

import re
from typing import List, Dict, Optional, Tuple, Any

# Global LanguageTool instance to avoid expensive re-initialization
_language_tool: Any = None


def _get_language_tool(language: str = "en-US") -> Any:
    """
    Get or initialize LanguageTool instance with lazy loading.

    LanguageTool initialization is expensive (~2-3 seconds), so we cache
    a single instance globally and reuse it across all requests.

    Args:
        language: Language code for grammar checking (default: en-US)

    Returns:
        LanguageTool instance
    """
    global _language_tool

    if _language_tool is None:
        import language_tool_python  # type: ignore
        _language_tool = language_tool_python.LanguageTool(language)
        print(f"✓ Loaded LanguageTool for {language}")

    return _language_tool


def grammar_errors(text: str, language: str = "en-US") -> Tuple[int, List[Dict[str, str]]]:
    """
    Detect grammar and spelling errors in text using LanguageTool.

    This function uses LanguageTool's rule-based grammar checker to identify:
    - Grammar mistakes
    - Spelling errors
    - Punctuation issues
    - Style suggestions

    Args:
        text: Input text to analyze
        language: Language code for grammar checking (default: en-US)

    Returns:
        tuple: (error_count, error_details) where:
            - error_count (int): Total number of errors detected
            - error_details (List[Dict]): List of error objects with keys:
                - message: Error description
                - rule_id: LanguageTool rule identifier
                - context: Text context around the error
                - offset: Character position of error
                - length: Length of error span
                - replacements: Suggested corrections (if available)

    Example:
        >>> count, errors = grammar_errors("She dont like apples.")
        >>> print(count)
        1
        >>> print(errors[0]["message"])
        "Use 'doesn't' instead of 'dont'"
    """
    if not text or not text.strip():
        return 0, []

    try:
        tool: Any = _get_language_tool(language)
        matches: Any = tool.check(text)

        # Extract relevant error information
        error_details: List[Dict[str, Any]] = []
        match: Any
        for match in matches:
            error_details.append({
                "message": str(match.message),
                "rule_id": str(match.ruleId),
                "context": str(match.context),
                "offset": int(match.offset),
                "length": int(match.errorLength),
                "replacements": match.replacements[:3] if match.replacements else []
            })

        return len(matches), error_details

    except Exception as e:
        print(f"⚠ Grammar check failed: {str(e)}")
        return 0, []


def filler_count(text: str) -> Tuple[int, List[str]]:
    """
    Count filler words and verbal disfluencies in text.

    Detects common filler words used in spoken language that indicate
    hesitation or lack of fluency. Includes both single-word fillers
    (um, uh) and multi-word phrases (you know, I mean).

    Args:
        text: Input text to analyze (typically from ASR transcript)

    Returns:
        tuple: (filler_count, filler_list) where:
            - filler_count (int): Total number of filler occurrences
            - filler_list (List[str]): List of detected filler instances

    Example:
        >>> count, fillers = filler_count("Um, I think, you know, it's like really good.")
        >>> print(count)
        3
        >>> print(fillers)
        ['um', 'you know', 'like']
    """
    if not text or not text.strip():
        return 0, []

    # Comprehensive list of common fillers in spoken English
    # Order matters: check multi-word phrases first to avoid partial matches
    filler_patterns = [
        # Multi-word fillers
        r'\byou know\b',
        r'\bi mean\b',
        r'\bkind of\b',
        r'\bkinda\b',
        r'\bsort of\b',
        r'\bsorta\b',
        r'\byou see\b',
        r'\blet me see\b',
        r'\blet\'s see\b',

        # Single-word fillers
        r'\bum+\b',           # um, umm, ummm
        r'\buh+\b',           # uh, uhh, uhhh
        r'\berm+\b',          # erm, ermm
        r'\bhmm+\b',          # hmm, hmmm
        r'\bah+\b',           # ah, ahh
        r'\boh+\b',           # oh, ohh
        r'\blike\b',          # like (when used as filler)
        r'\bbasically\b',
        r'\bactually\b',
        r'\bliterally\b',
        r'\bwell\b',          # well (at start of thought)
        r'\bso\b',            # so (as discourse marker)
        r'\bjust\b',          # just (as hedge)
    ]

    # Normalize text: lowercase and clean extra spaces
    normalized_text = re.sub(r'\s+', ' ', text.lower().strip())

    detected_fillers: List[str] = []

    # Search for each filler pattern
    for pattern in filler_patterns:
        matches: List[str] = re.findall(pattern, normalized_text, flags=re.IGNORECASE)
        detected_fillers.extend(matches)

    return len(detected_fillers), detected_fillers


def words_per_minute(word_count: int, duration_sec: float) -> Optional[float]:
    """
    Calculate speaking rate in words per minute (WPM).

    WPM is a key fluency metric:
    - Native speakers: typically 120-150 WPM in conversational speech
    - Non-native/learners: often 80-120 WPM
    - Very slow speech: < 80 WPM may indicate low proficiency or hesitation
    - Very fast speech: > 180 WPM may indicate nervousness or reduced clarity

    Args:
        word_count: Number of words spoken
        duration_sec: Audio duration in seconds

    Returns:
        float: Words per minute, rounded to 2 decimals
        None: If duration is missing, zero, or invalid

    Example:
        >>> wpm = words_per_minute(word_count=50, duration_sec=30.0)
        >>> print(wpm)
        100.0
    """
    if duration_sec <= 0:
        return None

    if word_count < 0:
        return None

    # Convert to WPM: (words / seconds) * 60
    wpm = (word_count / duration_sec) * 60.0

    return round(wpm, 2)


def sentence_stats(text: str) -> Dict[str, Any]:
    """
    Calculate basic sentence-level statistics.

    Provides metrics about sentence structure and complexity:
    - Total sentence count
    - Average sentence length (in words)
    - Shortest and longest sentences

    These metrics can indicate language proficiency:
    - Very short sentences: may indicate limited vocabulary or complexity
    - Very long sentences: may indicate run-on issues or lack of punctuation
    - Good variation: indicates natural, proficient language use

    Args:
        text: Input text to analyze

    Returns:
        dict: Sentence statistics with keys:
            - sentence_count (int): Total number of sentences
            - avg_sentence_length (float): Average words per sentence
            - min_sentence_length (int): Shortest sentence in words
            - max_sentence_length (int): Longest sentence in words

    Example:
        >>> stats = sentence_stats("Hello. This is a test. It works well.")
        >>> print(stats["sentence_count"])
        3
        >>> print(stats["avg_sentence_length"])
        2.33
    """
    if not text or not text.strip():
        return {
            "sentence_count": 0,
            "avg_sentence_length": 0.0,
            "min_sentence_length": 0,
            "max_sentence_length": 0
        }

    # Split text into sentences using common sentence delimiters
    # Handle periods, question marks, exclamation marks
    sentences = re.split(r'[.!?]+', text)

    # Filter out empty sentences and strip whitespace
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return {
            "sentence_count": 0,
            "avg_sentence_length": 0.0,
            "min_sentence_length": 0,
            "max_sentence_length": 0
        }

    # Calculate word count for each sentence
    sentence_lengths = [len(s.split()) for s in sentences]

    return {
        "sentence_count": len(sentences),
        "avg_sentence_length": round(sum(sentence_lengths) / len(sentence_lengths), 2),
        "min_sentence_length": min(sentence_lengths),
        "max_sentence_length": max(sentence_lengths)
    }


def normalize_transcript(text: str) -> str:
    """
    Clean and normalize ASR transcript for analysis.

    Performs light cleaning to standardize text from ASR output:
    - Removes excessive whitespace
    - Normalizes punctuation spacing
    - Strips leading/trailing whitespace
    - Preserves original casing and punctuation for grammar checking

    Args:
        text: Raw ASR transcript

    Returns:
        str: Normalized text ready for feature extraction

    Example:
        >>> normalized = normalize_transcript("  Hello   world  .  ")
        >>> print(normalized)
        "Hello world."
    """
    if not text:
        return ""

    # Remove excessive whitespace (multiple spaces/tabs/newlines)
    text = re.sub(r'\s+', ' ', text)

    # Fix spacing around punctuation (e.g., "word ." -> "word.")
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)

    text = text.strip()

    return text

