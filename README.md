# Grammar Scoring Engine from Voice Samples

**SHL Research Intern (AI) – Technical Assessment**

---

##  Project Overview

This project implements an **AI-powered grammar and fluency scoring system** for spoken English audio samples. It is designed to support **SHL's mission of building automated, objective assessments** for talent evaluation. The system takes raw audio input, transcribes it using state-of-the-art ASR (Whisper), extracts linguistic features (grammar errors, filler words, speaking rate), and produces a **standardized 0–100 proficiency score** with detailed feedback.

**Pipeline:** `Audio → ASR → Grammar & Fluency Analysis → Weighted Scoring → Explainable Output`

This work addresses real-world challenges in automated speech assessment: **accuracy, fairness, interpretability, and scalability**.

---

##  Why This Project Matters

- **SHL Context**: SHL builds AI-driven assessments to evaluate candidates' skills objectively. Spoken language proficiency is a critical dimension in many roles (customer service, teaching, communication-heavy positions).

- **Automated Speech Scoring**: Manual evaluation of speech is time-consuming and subjective. This system provides **consistent, scalable, and explainable scoring** grounded in linguistic research.

- **Research Relevance**: Combines **NLP** (grammar checking), **speech processing** (ASR), **psychometrics** (scoring calibration), and **AI ethics** (fairness across accents and dialects).

- **Practical Impact**: Can be deployed as a **REST API** for integration into larger assessment platforms or used as a **research tool** for analyzing speech datasets.

---

##  System Architecture

```
┌─────────────┐
│ Audio File  │
│  (.wav/.mp3)│
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│   Whisper ASR       │
│  (faster-whisper)   │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│   Transcription     │
│   (text + timing)   │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  Feature Extraction             │
│  • Grammar errors (LanguageTool)│
│  • Filler words (regex)         │
│  • WPM (fluency)                │
│  • WER (if reference given)     │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│   Scoring Engine                │
│   • Normalize penalties [0,1]   │
│   • Weighted final score        │
│   • Generate explanation        │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│   Output                        │
│   • JSON (API)                  │
│   • Structured report (Notebook)│
│   • Score + breakdown + feedback│
└─────────────────────────────────┘
```

---

##  Features

 **Whisper-based ASR**: Automatic speech recognition using OpenAI's Whisper (small model) with `faster-whisper` optimization  
 **Grammar Error Detection**: Rule-based analysis via LanguageTool with detailed error messages and suggestions  
 **Filler Word Detection**: Pattern matching for 15+ common fillers (um, uh, like, you know, etc.)  
 **Fluency Metrics**: Words per minute (WPM) calculation with ideal range benchmarks (110-170 WPM)  
 **WER (Optional)**: Word Error Rate calculation when reference transcript is provided  
 **Weighted Scoring**: Research-backed weights (Grammar 35%, Fillers 25%, WER 20%, Fluency 20%)  
 **Explainability**: Point-deduction breakdown for each component  
 **REST API**: FastAPI endpoint (`POST /score`) for production integration  
 **Jupyter Demo**: Interactive notebook for end-to-end pipeline exploration  
 **Unit Tests**: 34 tests covering scoring logic with pytest

---

##  Repository Structure

```
shl-grammar-scorer/
│
├── app/
│   ├── main.py              # FastAPI application with /score endpoint
│   ├── asr.py               # Whisper ASR with lazy model loading
│   ├── text_features.py     # Grammar, filler, WPM extraction
│   ├── scoring.py           # Normalization + final score calculation
│   ├── schemas.py           # Pydantic models for API responses
│   └── utils.py             # (Reserved for future utilities)
│
├── notebooks/
│   └── demo.ipynb           # End-to-end demo without API
│
├── tests/
│   └── test_scorer.py       # Unit tests for scoring logic (34 tests)
│
├── data/
│   └── sample_audio/        # Sample WAV/MP3 files for testing
│       └── sample1.wav      # (User-provided audio samples)
│
├── reports/
│   └── research_report.md   # (To be added: methodology, findings, evaluation)
│
├── requirements.txt         # Python dependencies
├── README.md                # This file
└── .gitignore               # Exclude models, temp files, etc.
```

---

##  How to Run (API Mode)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: For CPU-only PyTorch (faster installation):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 2. Start the API Server

```bash
uvicorn app.main:app --reload
```

The API will be available at: `http://localhost:8000`

### 3. Send a Scoring Request

**Using curl:**
```bash
curl -X POST "http://localhost:8000/score" \
  -F "audio=@data/sample_audio/sample1.wav" \
  -F "reference_transcript=Hello, my name is John."
```

**Using Python `requests`:**
```python
import requests

with open("data/sample_audio/sample1.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/score",
        files={"audio": f},
        data={"reference_transcript": "Optional reference text"}
    )

print(response.json())
```

### 4. View API Documentation

Visit: `http://localhost:8000/docs` (Swagger UI)

---

##  How to Run (Notebook Demo)

### 1. Install Dependencies (if not already done)

```bash
pip install -r requirements.txt
pip install jupyter  # If not installed
```

### 2. Launch Jupyter Notebook

```bash
jupyter notebook notebooks/demo.ipynb
```

### 3. Run All Cells

The notebook demonstrates:
- ASR transcription
- Feature extraction
- Scoring calculation
- Visualization (score breakdown, penalty chart)
- Batch evaluation (if multiple audio files provided)

**Note**: Place audio files in `data/sample_audio/` directory before running.

---

##  Tests

Run unit tests with pytest:

```bash
# Run all tests
pytest -v

# Run with coverage
pytest --cov=app tests/

# Run specific test file
pytest tests/test_scorer.py -v
```

**Test Coverage**: 34 unit tests covering:
- Grammar error normalization
- Filler word normalization
- WER normalization
- Fluency penalty calculation
- Final score calculation
- Weight validation
- Boundary condition handling

---

##  Example Output

**API Response (`POST /score`):**

```json
{
  "asr": {
    "transcript": "Um, hello. My name is John and, you know, I am applying for this position.",
    "word_count": 15,
    "duration_sec": 7.5,
    "language": "en"
  },
  "metrics": {
    "grammar_errors": 0,
    "fillers": 2,
    "wer": null,
    "wpm": 120.0,
    "normalized": {
      "grammar": 0.0,
      "fillers": 0.1667,
      "wer": 0.0,
      "fluency": 0.0
    },
    "final_score": 95.83
  },
  "grammar_details": [],
  "filler_words": ["um", "you know"],
  "explanation": "Score: 95.83/100 | Grammar: -0.0 pts | Fillers: -4.17 pts | WER: -0.0 pts | Fluency: -0.0 pts",
  "model_version": "faster-whisper-small",
  "generated_at": "2025-11-08T14:32:10.123456"
}
```

---

##  Scoring Methodology

### Weighted Components

| Component | Weight | Rationale |
|-----------|--------|-----------|
| **Grammar** | 35% | Most critical indicator of language proficiency. Grammar errors directly reflect understanding of language rules. |
| **Fillers** | 25% | Strong indicator of fluency and confidence. Excessive fillers suggest hesitation or lack of vocabulary. |
| **WER (Accuracy)** | 20% | Measures pronunciation clarity when reference transcript is available. Optional component. |
| **Fluency (WPM)** | 20% | Speaking rate indicates comfort and automaticity. Optimal range: 110-170 WPM. |

### Normalization Formula

Each component is converted to a **penalty value** ∈ [0, 1]:

#### Grammar Error Rate (GER)
```
GER = (error_count / word_count) × 100
Penalty = min(1.0, GER / 12.0)
```
- 0 errors → 0.0 penalty (perfect)
- 12+ errors per 100 words → 1.0 penalty (maximum)

#### Filler Rate
```
Filler_Rate = (filler_count / word_count) × 100
Penalty = min(1.0, Filler_Rate / 8.0)
```
- 0 fillers → 0.0 penalty
- 8+ fillers per 100 words → 1.0 penalty

#### WER (Word Error Rate)
```
Penalty = min(1.0, WER / 0.35)
```
- 0% WER → 0.0 penalty (perfect accuracy)
- 35%+ WER → 1.0 penalty

#### Fluency (WPM)
```
If 110 ≤ WPM ≤ 170: Penalty = 0.0
If WPM < 110: Penalty = (110 - WPM) / (110 - 60)
If WPM > 170: Penalty = (WPM - 170) / (220 - 170)
```

### Final Score Calculation

```
Total_Penalty = (
    Grammar_Penalty × 0.35 +
    Filler_Penalty × 0.25 +
    WER_Penalty × 0.20 +
    Fluency_Penalty × 0.20
)

Final_Score = 100 - (Total_Penalty × 100)
```

Scores are clamped to [0, 100] and rounded to 2 decimal places.

---

##  Limitations & Ethical Considerations

### Current Limitations

1. **ASR Accuracy**: Whisper may introduce transcription errors, especially with:
   - Heavy accents
   - Background noise
   - Domain-specific terminology
   - Code-switching (multilingual speakers)

2. **Grammar Checker Constraints**:
   - LanguageTool is rule-based and may miss context-dependent errors
   - May flag valid colloquialisms as errors
   - Limited support for creative language use

3. **Filler Word Detection**:
   - Some "fillers" (like "well", "so") serve as discourse markers
   - Context matters: "like" as a verb vs. filler
   - Frequency norms vary by culture and speaking style

4. **Fluency Oversimplification**:
   - WPM alone doesn't capture pause patterns, prosody, or stress
   - Fast speech ≠ fluent speech (may lack clarity)
   - Doesn't account for thinking pauses vs. hesitation

### Bias and Fairness Concerns

 **Non-native speakers**: May be unfairly penalized for grammatical structures or fillers that are artifacts of language transfer, not lack of proficiency.

 **Dialect variation**: AAVE (African American Vernacular English), regional dialects, and sociolects may trigger false grammar errors.

 **Accent bias in ASR**: Whisper performs better on mainstream American/British accents. Speakers with other accents may receive inaccurate transcriptions, leading to inflated error counts.

 **Cultural norms**: Speaking rate and filler use vary across cultures. A slower pace may indicate thoughtfulness, not disfluency.

### Recommendations for Fair Use

1. **Calibration**: Validate scoring against human expert ratings across diverse speaker populations
2. **Transparency**: Always show detailed breakdown (not just final score)
3. **Human-in-the-loop**: Use scores as decision support, not sole determinant
4. **Accent normalization**: Fine-tune ASR on diverse accent data
5. **Contextual evaluation**: Combine with other assessment modalities (written, situational)

### Future Work

- [ ] **Prosody analysis**: Incorporate pitch, stress, and intonation features
- [ ] **LLM-based grammar**: Use GPT/BERT for context-aware error detection
- [ ] **Pause detection**: Analyze silence patterns for true disfluency vs. strategic pauses
- [ ] **Multi-reference WER**: Support multiple valid reference transcripts
- [ ] **Fairness audits**: Test scoring consistency across demographic groups
- [ ] **Active learning**: Iteratively improve with human feedback

---

##  Author

**Battu Bharath Kumar**  
AI Research Intern Candidate | SHL Assessment  

 [LinkedIn](https://www.linkedin.com/in/battu-bharath-kumar/)  
 [GitHub](https://github.com/bharathbattu)  
 2200032221cser@gmail.com

---

##  License

This project is developed as part of a technical assessment for SHL.  
Code is provided under the **MIT License** for educational and evaluation purposes.

```
MIT License

Copyright (c) 2025 Battu Bharath Kumar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

##  Acknowledgments

- **OpenAI Whisper**: State-of-the-art ASR model
- **LanguageTool**: Open-source grammar checking
- **SHL**: For providing the opportunity to work on this meaningful problem
- **Research Community**: NLP and speech assessment researchers whose work informed the scoring methodology

---

**Built with Love for AI-powered talent assessment | November 2025**
