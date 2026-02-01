import os
import asyncio
from typing import List, Optional
from pydantic import BaseModel, Field
import openai
import json
from dotenv import load_dotenv

# Initialize early
# Search for .env in current and parent directories
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

# Configuration
API_KEY = os.getenv("GROQ_API_KEY") 
BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# Data Models
class PredictionContext(BaseModel):
    predicted_class: str
    confidence_score: float
    probabilities: dict[str, float]
    model_architecture: str

class ClinicalReport(BaseModel):
    patient_summary: str
    detailed_findings: str
    risk_assessment: str
    recommended_actions: List[str]
    questions_for_specialist: List[str]

async def generate_clinical_report(ctx: PredictionContext) -> ClinicalReport:
    """
    Final robust version: Direct OpenAI-style call with Pydantic parsing.
    """
    prompt = f"Patient MRI Analysis:\nStage: {ctx.predicted_class}\nConfidence: {ctx.confidence_score:.2%}\nArchitecture: {ctx.model_architecture}"
    system = """You are a Neurologist. Return ONLY a JSON object with these keys:
    "patient_summary" (string), 
    "detailed_findings" (string), 
    "risk_assessment" (string), 
    "recommended_actions" (list of strings), 
    "questions_for_specialist" (list of strings)."""

    try:
        client = openai.AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        return ClinicalReport(**data)
    except Exception as e:
        print(f"Direct API Error: {e}")
        return ClinicalReport(
            patient_summary="Dynamic report generation unavailable.",
            detailed_findings=f"The model predicted {ctx.predicted_class} with {ctx.confidence_score:.1%} confidence.",
            risk_assessment="Awaiting specialist review.",
            recommended_actions=["Manual MRI review recommended."],
            questions_for_specialist=["Is the hippocampus atrophy consistent with the stage?"]
        )

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    ctx = PredictionContext(predicted_class="ModerateDemented", confidence_score=0.9, probabilities={}, model_architecture="Hybrid")
    res = asyncio.run(generate_clinical_report(ctx))
    print(res.model_dump_json(indent=2))
