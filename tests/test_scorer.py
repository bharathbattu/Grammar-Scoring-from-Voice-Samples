"""
Unit tests for scoring logic.

Tests the core scoring functions in app/scoring.py to ensure:
- Correct normalization of penalties
- Proper boundary conditions
- Weight calculations
- Score clamping

Run with: pytest tests/test_scorer.py -v
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.scoring import (
    clamp_01,
    normalize_grammar_errors,
    normalize_fillers,
    normalize_wer,
    fluency_penalty,
    calculate_final_score,
    WEIGHT_GRAMMAR,
    WEIGHT_FILLERS,
    WEIGHT_WER,
    WEIGHT_FLUENCY,
)


class TestNormalizeGrammarErrors:
    """Test grammar error normalization."""

    def test_zero_errors_returns_zero_penalty(self):
        """Perfect grammar should result in zero penalty."""
        penalty = normalize_grammar_errors(error_count=0, word_count=100)
        assert penalty == 0.0

    def test_max_errors_returns_one_penalty(self):
        """12+ errors per 100 words should result in maximum penalty."""
        # Exactly at threshold
        penalty = normalize_grammar_errors(error_count=12, word_count=100)
        assert penalty == 1.0

        # Beyond threshold
        penalty = normalize_grammar_errors(error_count=20, word_count=100)
        assert penalty == 1.0

    def test_intermediate_errors_proportional(self):
        """6 errors per 100 words should be ~0.5 penalty."""
        penalty = normalize_grammar_errors(error_count=6, word_count=100)
        assert 0.45 <= penalty <= 0.55  # Allow small floating point variance

    def test_scales_with_word_count(self):
        """Same error rate should produce same penalty regardless of word count."""
        # 3 errors in 50 words = 6 errors per 100 words
        penalty_1 = normalize_grammar_errors(error_count=3, word_count=50)

        # 6 errors in 100 words = 6 errors per 100 words
        penalty_2 = normalize_grammar_errors(error_count=6, word_count=100)

        assert abs(penalty_1 - penalty_2) < 0.01

    def test_zero_words_returns_zero_penalty(self):
        """Empty transcript should not penalize."""
        penalty = normalize_grammar_errors(error_count=5, word_count=0)
        assert penalty == 0.0


class TestNormalizeFillers:
    """Test filler word normalization."""

    def test_zero_fillers_returns_zero_penalty(self):
        """No fillers should result in zero penalty."""
        penalty = normalize_fillers(filler_count=0, word_count=100)
        assert penalty == 0.0

    def test_max_fillers_returns_one_penalty(self):
        """8+ fillers per 100 words should result in maximum penalty."""
        # Exactly at threshold
        penalty = normalize_fillers(filler_count=8, word_count=100)
        assert penalty == 1.0

        # Beyond threshold
        penalty = normalize_fillers(filler_count=15, word_count=100)
        assert penalty == 1.0

    def test_intermediate_fillers_proportional(self):
        """4 fillers per 100 words should be ~0.5 penalty."""
        penalty = normalize_fillers(filler_count=4, word_count=100)
        assert 0.45 <= penalty <= 0.55

    def test_scales_with_word_count(self):
        """Same filler rate should produce same penalty regardless of word count."""
        # 2 fillers in 50 words = 4 per 100 words
        penalty_1 = normalize_fillers(filler_count=2, word_count=50)

        # 4 fillers in 100 words = 4 per 100 words
        penalty_2 = normalize_fillers(filler_count=4, word_count=100)

        assert abs(penalty_1 - penalty_2) < 0.01

    def test_zero_words_returns_zero_penalty(self):
        """Empty transcript should not penalize."""
        penalty = normalize_fillers(filler_count=3, word_count=0)
        assert penalty == 0.0


class TestNormalizeWER:
    """Test WER (Word Error Rate) normalization."""

    def test_zero_wer_returns_zero_penalty(self):
        """Perfect accuracy should result in zero penalty."""
        penalty = normalize_wer(wer=0.0)
        assert penalty == 0.0

    def test_max_wer_returns_one_penalty(self):
        """WER >= 0.35 should result in maximum penalty."""
        # Exactly at threshold
        penalty = normalize_wer(wer=0.35)
        assert penalty == 1.0

        # Beyond threshold
        penalty = normalize_wer(wer=0.50)
        assert penalty == 1.0

    def test_intermediate_wer_proportional(self):
        """WER of 0.175 (half of max) should be ~0.5 penalty."""
        penalty = normalize_wer(wer=0.175)
        assert 0.45 <= penalty <= 0.55

    def test_none_wer_returns_zero_penalty(self):
        """No reference transcript (None) should not penalize."""
        penalty = normalize_wer(wer=None)
        assert penalty == 0.0

    def test_negative_wer_returns_zero_penalty(self):
        """Invalid negative WER should safely return zero."""
        penalty = normalize_wer(wer=-0.1)
        assert penalty == 0.0


class TestFluencyPenalty:
    """Test fluency (WPM) penalty calculation."""

    def test_ideal_wpm_returns_zero_penalty(self):
        """WPM within ideal range (110-170) should have no penalty."""
        assert fluency_penalty(wpm=110.0) == 0.0
        assert fluency_penalty(wpm=140.0) == 0.0
        assert fluency_penalty(wpm=170.0) == 0.0

    def test_very_slow_wpm_returns_high_penalty(self):
        """WPM <= 60 should result in maximum penalty."""
        penalty = fluency_penalty(wpm=60.0)
        assert penalty == 1.0

        penalty = fluency_penalty(wpm=40.0)
        assert penalty == 1.0

    def test_very_fast_wpm_returns_high_penalty(self):
        """WPM >= 220 should result in maximum penalty."""
        penalty = fluency_penalty(wpm=220.0)
        assert penalty == 1.0

        penalty = fluency_penalty(wpm=250.0)
        assert penalty == 1.0

    def test_moderately_slow_wpm_has_moderate_penalty(self):
        """WPM of 90 should have some penalty but not maximum."""
        penalty = fluency_penalty(wpm=90.0)
        assert 0.2 <= penalty <= 0.6

    def test_moderately_fast_wpm_has_moderate_penalty(self):
        """WPM of 190 should have some penalty but not maximum."""
        penalty = fluency_penalty(wpm=190.0)
        assert 0.2 <= penalty <= 0.6

    def test_none_wpm_returns_zero_penalty(self):
        """Missing WPM data should not penalize."""
        penalty = fluency_penalty(wpm=None)
        assert penalty == 0.0

    def test_zero_wpm_returns_zero_penalty(self):
        """Invalid zero WPM should safely return zero."""
        penalty = fluency_penalty(wpm=0.0)
        assert penalty == 0.0

    def test_negative_wpm_returns_zero_penalty(self):
        """Invalid negative WPM should safely return zero."""
        penalty = fluency_penalty(wpm=-50.0)
        assert penalty == 0.0


class TestCalculateFinalScore:
    """Test final score calculation and weighting."""

    def test_perfect_score_with_zero_penalties(self):
        """All zero penalties should result in score of 100."""
        score = calculate_final_score(
            grammar_penalty=0.0,
            filler_penalty=0.0,
            wer_penalty=0.0,
            fluency_pen=0.0
        )
        assert score == 100.0

    def test_worst_score_with_max_penalties(self):
        """All maximum penalties should result in score of 0."""
        score = calculate_final_score(
            grammar_penalty=1.0,
            filler_penalty=1.0,
            wer_penalty=1.0,
            fluency_pen=1.0
        )
        assert score == 0.0

    def test_score_never_exceeds_bounds(self):
        """Score should always be clamped to [0, 100]."""
        # Test with extreme values
        score = calculate_final_score(
            grammar_penalty=2.0,  # Beyond max
            filler_penalty=2.0,
            wer_penalty=2.0,
            fluency_pen=2.0
        )
        assert 0.0 <= score <= 100.0

        # Test with negative values
        score = calculate_final_score(
            grammar_penalty=-1.0,
            filler_penalty=-1.0,
            wer_penalty=-1.0,
            fluency_pen=-1.0
        )
        assert 0.0 <= score <= 100.0

    def test_grammar_weight_dominates(self):
        """Grammar penalty should have largest impact on score (35% weight)."""
        # Only grammar penalty
        score_grammar = calculate_final_score(
            grammar_penalty=1.0,
            filler_penalty=0.0,
            wer_penalty=0.0,
            fluency_pen=0.0
        )

        # Only filler penalty
        score_filler = calculate_final_score(
            grammar_penalty=0.0,
            filler_penalty=1.0,
            wer_penalty=0.0,
            fluency_pen=0.0
        )

        # Grammar should reduce score more than fillers
        assert score_grammar < score_filler

        # Check actual weight impact
        grammar_deduction = 100.0 - score_grammar
        filler_deduction = 100.0 - score_filler

        assert abs(grammar_deduction - 35.0) < 0.1  # Grammar weight is 35%
        assert abs(filler_deduction - 25.0) < 0.1   # Filler weight is 25%

    def test_weights_are_properly_applied(self):
        """Verify that weights are correctly applied to each component."""
        # Half penalty on grammar only
        score = calculate_final_score(
            grammar_penalty=0.5,
            filler_penalty=0.0,
            wer_penalty=0.0,
            fluency_pen=0.0
        )
        expected_deduction = 0.5 * WEIGHT_GRAMMAR * 100
        expected_score = 100.0 - expected_deduction
        assert abs(score - expected_score) < 0.1

    def test_mixed_penalties_realistic_scenario(self):
        """Test realistic scenario with mixed penalties."""
        # Moderate proficiency speaker:
        # - Some grammar errors (0.3 penalty)
        # - Few fillers (0.2 penalty)
        # - Good accuracy (0.1 WER penalty)
        # - Good fluency (0.0 penalty)
        score = calculate_final_score(
            grammar_penalty=0.3,
            filler_penalty=0.2,
            wer_penalty=0.1,
            fluency_pen=0.0
        )

        # Expected deductions:
        # Grammar: 0.3 * 35 = 10.5
        # Fillers: 0.2 * 25 = 5.0
        # WER: 0.1 * 20 = 2.0
        # Fluency: 0.0 * 20 = 0.0
        # Total: 17.5 points deducted
        # Score: 100 - 17.5 = 82.5

        assert 81.0 <= score <= 84.0  # Allow small variance

    def test_score_precision_two_decimals(self):
        """Final score should be rounded to 2 decimal places."""
        score = calculate_final_score(
            grammar_penalty=0.333,
            filler_penalty=0.333,
            wer_penalty=0.333,
            fluency_pen=0.333
        )

        # Check that score has at most 2 decimal places
        score_str = str(score)
        if '.' in score_str:
            decimals = len(score_str.split('.')[1])
            assert decimals <= 2


class TestClamp01:
    """Test utility function for clamping values."""

    def test_clamp_below_zero(self):
        """Values below 0 should be clamped to 0."""
        assert clamp_01(-0.5) == 0.0
        assert clamp_01(-100.0) == 0.0

    def test_clamp_above_one(self):
        """Values above 1 should be clamped to 1."""
        assert clamp_01(1.5) == 1.0
        assert clamp_01(100.0) == 1.0

    def test_clamp_within_range(self):
        """Values within [0, 1] should remain unchanged."""
        assert clamp_01(0.0) == 0.0
        assert clamp_01(0.5) == 0.5
        assert clamp_01(1.0) == 1.0


class TestWeightSum:
    """Test that scoring weights are properly configured."""

    def test_weights_sum_to_one(self):
        """All weights should sum to 1.0 for proper normalization."""
        total = WEIGHT_GRAMMAR + WEIGHT_FILLERS + WEIGHT_WER + WEIGHT_FLUENCY
        assert abs(total - 1.0) < 0.001  # Allow tiny floating point error


# Note: Additional tests will be added later for:
# - text_features.py (grammar_errors, filler_count, etc.)
# - FastAPI endpoint integration tests
# - End-to-end pipeline tests with real audio samples
