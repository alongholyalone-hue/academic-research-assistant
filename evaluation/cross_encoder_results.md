# Evaluation Results

This evaluation uses a controlled four-page synthetic PDF.

## Summary

- Total questions: 10
- Retrieval Hit@3: 100.0%
- Answer accuracy: 87.5%
- Citation accuracy: 100.0%
- Unsupported-question refusal accuracy: 100.0%
- Overall success rate: 90.0%

## Detailed Results

| ID | Expected | Predicted | Retrieved pages | Citation page | Passed |
|---|---|---|---|---|---|
| q1 | 8:00 AM | 8:00 AM | [1, 2, 4] | 1 | Yes |
| q2 | 9:00 PM | 9:00 PM | [1, 2, 4] | 1 | Yes |
| q3 | fifteen | fifteen | [2, 1, 4] | 2 | Yes |
| q4 | Tuesdays and Thursdays | Tuesdays and Thursdays | [2, 1, 4] | 2 | Yes |
| q5 | fairness | fairness, transparency, and human oversight | [3, 4, 1] | 3 | Yes |
| q6 | Python | Python | [4, 3, 1] | 4 | Yes |
| q7 | FastAPI | FastAPI | [4, 1, 2] | 4 | Yes |
| q8 | Streamlit | FastAPI | [4, 2, 3] | 4 | No |
| q9 | — | The uploaded document does not provide enough evidence to answer this question. | [1, 2, 4] | — | Yes |
| q10 | — | The uploaded document does not provide enough evidence to answer this question. | [1, 4, 2] | — | Yes |

## Interpretation

The scores measure performance on a small controlled evaluation set and should not be interpreted as general real-world accuracy.

The evaluation is intended to identify retrieval, answer-extraction, citation, and refusal weaknesses that require further testing.
