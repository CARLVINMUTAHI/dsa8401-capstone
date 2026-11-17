# AI Use Disclosure
### DSA 8401 Applied Machine Learning — Capstone Project
**Author:** Carlvin Mutahi · 101055

---

In accordance with the course policy on AI assistance, I disclose the following use of AI tools in this capstone project:

## AI Tools Used

**Claude (Anthropic)** was used as a coding assistant during the development of this project.

## Specific Uses

| Area | AI Assistance | Student Contribution |
|---|---|---|
| Data pipeline (`01_pipeline.py`) | Helped structure the script layout and suggested DHS variable names for WHO anthropometric flag bounds | All feature engineering decisions, variable selection and KDHS-specific logic were defined by the student based on domain knowledge of DHS survey structure |
| Model training (`02_train.py`) | Assisted with the imbalanced-learn pipeline structure and SHAP integration code | Model selection rationale, hyperparameter choices, evaluation metric prioritisation (AUC-ROC + Recall), and technique justification were the student's own analytical decisions |
| FastAPI service (`app/main.py`) | Helped with Pydantic schema structure and endpoint design | API design decisions, field validation ranges, and response schema were defined by the student |
| Docker configuration | Suggested multi-stage build pattern | Reviewed and adapted to project requirements |
| README.md | Assisted with formatting | All technical content, architecture decisions and descriptions authored by student |

## What AI Did NOT Do

- AI did not select the research problem or dataset
- AI did not make decisions about which ML algorithms to use or why
- AI did not write the technique justification (the reasoning in `02_train.py` docstring reflects the student's own analytical position)
- AI did not interpret results or draw conclusions
- AI did not write the research proposal or presentation materials

## Student Declaration

All design decisions, analytical choices, interpretations and conclusions in this project are my own. AI assistance was used as a productivity tool, equivalent to using Stack Overflow or documentation, and did not replace my own understanding or judgement.

**Carlvin Mutahi**  
101055  
MSc Data Science and Analytics  
Strathmore University  
November 2026
