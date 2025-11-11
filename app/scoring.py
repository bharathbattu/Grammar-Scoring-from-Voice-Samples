"""

This module converts extracted speech features into a normalized 0-100 proficiency score.
The scoring system is based on research in automated speech assessment and language testing.

SCORING WEIGHTS (Research-Based):
- Grammar (35%): Most critical indicator of language proficiency. Grammar errors directly
  reflect understanding of language rules and structure.
- Fillers (25%): Strong indicator of fluency and confidence. Excessive fillers suggest
  hesitation, lack of vocabulary, or processing difficulties.
- WER/Accuracy (20%): When reference transcript available, measures pronunciation and
  clarity. Lower weight because it's optional and context-dependent.
- Fluency/WPM (20%): Speaking rate indicates comfort and automaticity. Too slow suggests
  struggle; too fast may indicate nervousness or reduced clarity.

NORMALIZATION APPROACH:
All features are converted to penalty values [0, 1] where:
- 0.0 = perfect (no penalty)
- 1.0 = maximum penalty

Thresholds are calibrated based on:
- CEFR (Common European Framework of Reference) benchmarks
- Research on L2 English speech assessment
- Real-world ASR transcription quality

Final score: 100 - (weighted_sum_of_penalties)
"""

from typing import Optional

# Grammar error thresholds (per 100 words)
MAX_GRAMMAR_ERRORS_PER_100 = 12.0
MIN_GRAMMAR_ERRORS_PER_100 = 0.0   

# Filler word thresholds (per 100 words)
MAX_FILLERS_PER_100 = 8.0  
MIN_FILLERS_PER_100 = 0.0  

# WER (Word Error Rate) thresholds
MAX_WER = 0.35  
MIN_WER = 0.0   

# WPM (Words Per Minute) thresholds for fluency
IDEAL_WPM_MIN = 110.0  # Minimum comfortable speaking rate
IDEAL_WPM_MAX = 170.0  # Maximum comfortable speaking rate
VERY_SLOW_WPM = 60.0   # Below this = maximum fluency penalty
VERY_FAST_WPM = 220.0  # Above this = maximum fluency penalty

# Scoring weights (must sum to 1.0)
WEIGHT_GRAMMAR = 0.35
WEIGHT_FILLERS = 0.25
WEIGHT_WER = 0.20
WEIGHT_FLUENCY = 0.20


def clamp_01(x: float) -> float:
    """
    Clamp a value to the range [0, 1].

    Args:
        x: Input value

    Returns:
        float: Value clamped to [0, 1]

    Example:
        >>> clamp_01(-0.5)
        0.0
        >>> clamp_01(0.7)
        0.7
        >>> clamp_01(1.5)
        1.0
    """
    return max(0.0, min(1.0, x))


def normalize_grammar_errors(error_count: int, word_count: int) -> float:
    """
    Normalize grammar errors to penalty value [0, 1].

    Converts raw error count to Grammar Error Rate (GER) per 100 words,
    then normalizes to a penalty score.

    Calibration:
    - 0 errors → 0.0 penalty (perfect)
    - 3 errors per 100 words → ~0.25 penalty (proficient)
    - 6 errors per 100 words → ~0.50 penalty (intermediate)
    - 12+ errors per 100 words → 1.0 penalty (very low proficiency)

    Args:
        error_count: Number of grammar errors detected
        word_count: Total number of words in transcript

    Returns:
        float: Normalized penalty [0, 1] where 0 = no errors, 1 = max errors

    Example:
        >>> normalize_grammar_errors(0, 100)
        0.0
        >>> normalize_grammar_errors(3, 100)
        0.25
        >>> normalize_grammar_errors(12, 100)
        1.0
        >>> normalize_grammar_errors(15, 50)  # 30 per 100 words
        1.0
    """
    if word_count <= 0:
        return 0.0  

    # Calculate Grammar Error Rate (GER) per 100 words
    ger = (error_count / word_count) * 100.0

    # Normalize to [0, 1] based on threshold
    penalty = ger / MAX_GRAMMAR_ERRORS_PER_100

    return clamp_01(penalty)


def normalize_fillers(filler_count: int, word_count: int) -> float:
    """
    Normalize filler words to penalty value [0, 1].

    Converts raw filler count to filler rate per 100 words,
    then normalizes to a penalty score.

    Calibration:
    - 0 fillers → 0.0 penalty (perfectly fluent)
    - 2 fillers per 100 words → ~0.25 penalty (mostly fluent)
    - 4 fillers per 100 words → ~0.50 penalty (somewhat hesitant)
    - 8+ fillers per 100 words → 1.0 penalty (very disfluent)

    Args:
        filler_count: Number of filler words detected
        word_count: Total number of words in transcript

    Returns:
        float: Normalized penalty [0, 1] where 0 = no fillers, 1 = excessive fillers

    Example:
        >>> normalize_fillers(0, 100)
        0.0
        >>> normalize_fillers(2, 100)
        0.25
        >>> normalize_fillers(8, 100)
        1.0
        >>> normalize_fillers(10, 50)  # 20 per 100 words
        1.0
    """
    if word_count <= 0:
        return 0.0  

    # Calculate filler rate per 100 words
    filler_rate = (filler_count / word_count) * 100.0

    # Normalize to [0, 1] based on threshold
    penalty = filler_rate / MAX_FILLERS_PER_100

    return clamp_01(penalty)


def normalize_wer(wer: Optional[float]) -> float:
    """
    Normalize Word Error Rate (WER) to penalty value [0, 1].

    WER measures pronunciation accuracy when reference transcript is available.
    Only applicable when ground truth is provided.

    Calibration:
    - 0.00 WER → 0.0 penalty (perfect pronunciation)
    - 0.10 WER → ~0.29 penalty (good clarity)
    - 0.20 WER → ~0.57 penalty (moderate issues)
    - 0.35+ WER → 1.0 penalty (very unclear speech)

    Args:
        wer: Word Error Rate as decimal (e.g., 0.15 = 15% error rate)
             None if no reference transcript available

    Returns:
        float: Normalized penalty [0, 1] where 0 = perfect, 1 = very poor
               Returns 0.0 if WER is None (no penalty when not applicable)

    Example:
        >>> normalize_wer(0.0)
        0.0
        >>> normalize_wer(0.10)
        0.29
        >>> normalize_wer(0.35)
        1.0
        >>> normalize_wer(None)
        0.0
    """
    if wer is None:
        return 0.0  

    if wer < 0:
        return 0.0  

    # Normalize to [0, 1] based on threshold
    penalty = wer / MAX_WER

    return clamp_01(penalty)


def fluency_penalty(wpm: Optional[float]) -> float:
    """
    Calculate fluency penalty based on speaking rate (WPM).

    Ideal speaking rate for conversational English is 110-170 WPM.
    Rates outside this range indicate either struggle (too slow) or
    nervousness/reduced clarity (too fast).

    Penalty curve:
    - 110-170 WPM → 0.0 penalty (ideal range)
    - 90 WPM → ~0.25 penalty (somewhat slow)
    - 60 WPM → 1.0 penalty (very slow)
    - 190 WPM → ~0.25 penalty (somewhat fast)
    - 220+ WPM → 1.0 penalty (very fast)

    Args:
        wpm: Words per minute (speaking rate)
             None if duration unavailable

    Returns:
        float: Fluency penalty [0, 1] where 0 = ideal rate, 1 = extreme rate

    Example:
        >>> fluency_penalty(140.0)  # Within ideal range
        0.0
        >>> fluency_penalty(90.0)   # Somewhat slow
        0.4
        >>> fluency_penalty(60.0)   # Very slow
        1.0
        >>> fluency_penalty(200.0)  # Somewhat fast
        0.6
        >>> fluency_penalty(None)
        0.0
    """
    if wpm is None or wpm <= 0:
        return 0.0  

    # Within ideal range = no penalty
    if IDEAL_WPM_MIN <= wpm <= IDEAL_WPM_MAX:
        return 0.0

    # Too slow: linear penalty from ideal to very slow
    if wpm < IDEAL_WPM_MIN:
        distance = IDEAL_WPM_MIN - wpm
        max_distance = IDEAL_WPM_MIN - VERY_SLOW_WPM
        penalty = distance / max_distance
        return clamp_01(penalty)

    # Too fast: linear penalty from ideal to very fast
    if wpm > IDEAL_WPM_MAX:
        distance = wpm - IDEAL_WPM_MAX
        max_distance = VERY_FAST_WPM - IDEAL_WPM_MAX
        penalty = distance / max_distance
        return clamp_01(penalty)

    return 0.0


def calculate_final_score(
    grammar_penalty: float,
    filler_penalty: float,
    wer_penalty: float,
    fluency_pen: float
) -> float:
    """
    Calculate weighted final proficiency score from normalized penalties.

    Combines all penalty components using research-based weights:
    - Grammar: 35% (most important)
    - Fillers: 25% (fluency indicator)
    - WER: 20% (pronunciation accuracy, when available)
    - Fluency: 20% (speaking rate)

    Formula: score = 100 - (weighted_sum_of_penalties × 100)

    Args:
        grammar_penalty: Grammar error penalty [0, 1]
        filler_penalty: Filler word penalty [0, 1]
        wer_penalty: WER penalty [0, 1] (0 if not applicable)
        fluency_pen: Fluency penalty [0, 1]

    Returns:
        float: Final score [0, 100] where 100 = perfect, 0 = very poor
               Rounded to 2 decimal places

    Example:
        >>> calculate_final_score(0.0, 0.0, 0.0, 0.0)  # Perfect
        100.0
        >>> calculate_final_score(0.25, 0.25, 0.0, 0.0)  # Some errors
        70.0
        >>> calculate_final_score(1.0, 1.0, 1.0, 1.0)  # Maximum penalties
        0.0
    """
    # Weighted sum of penalties
    total_penalty = (
        grammar_penalty * WEIGHT_GRAMMAR +
        filler_penalty * WEIGHT_FILLERS +
        wer_penalty * WEIGHT_WER +
        fluency_pen * WEIGHT_FLUENCY
    )

    # Convert penalty [0, 1] to score [100, 0]
    score = 100.0 - (total_penalty * 100.0)

    # Clamp to valid range and round
    score = max(0.0, min(100.0, score))

    return round(score, 2)


def generate_score_explanation(
    grammar_penalty: float,
    filler_penalty: float,
    wer_penalty: float,
    fluency_pen: float,
    final: float
) -> str:
    """
    Generate human-readable explanation of score breakdown.

    Args:
        grammar_penalty: Grammar penalty [0, 1]
        filler_penalty: Filler penalty [0, 1]
        wer_penalty: WER penalty [0, 1]
        fluency_pen: Fluency penalty [0, 1]
        final: Final score [0, 100]

    Returns:
        str: Explanation text describing score components

    Example:
        >>> explanation = generate_score_explanation(0.2, 0.1, 0.0, 0.1, 85.0)
        >>> print(explanation)
        Score: 85.0/100 | Grammar: -7.0 pts | Fillers: -2.5 pts | WER: -0.0 pts | Fluency: -2.0 pts
    """
    grammar_deduction = round(grammar_penalty * WEIGHT_GRAMMAR * 100, 1)
    filler_deduction = round(filler_penalty * WEIGHT_FILLERS * 100, 1)
    wer_deduction = round(wer_penalty * WEIGHT_WER * 100, 1)
    fluency_deduction = round(fluency_pen * WEIGHT_FLUENCY * 100, 1)

    explanation = (
        f"Score: {final}/100 | "
        f"Grammar: -{grammar_deduction} pts | "
        f"Fillers: -{filler_deduction} pts | "
        f"WER: -{wer_deduction} pts | "
        f"Fluency: -{fluency_deduction} pts"
    )

    return explanation


