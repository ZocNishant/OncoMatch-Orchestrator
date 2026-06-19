from typing import List

from pydantic import BaseModel, Field


# 1. Define the Strict Data Blueprint
# This acts as a contract. Any data entering our agent pipeline MUST look like this.
class PatientProfile(BaseModel):
    cancer_type: str = Field(description="The primary type of cancer diagnosed.")
    stage: str = Field(description="The clinical stage of the cancer (e.g., Stage IV).")
    mutations: List[str] = Field(
        default=[], description="Genetic mutations identified (e.g., EGFR, KRAS)."
    )
    prior_treatments: List[str] = Field(
        default=[], description="Treatments the patient has already received."
    )


# 2. Raw, Unstructured Doctor Notes (The Chaos)
raw_doctor_notes = """
Patient is a 54-year-old male presenting with a persistent cough. 
Biopsy confirms advanced Non-Small Cell Lung Cancer, specifically Stage IV. 
Molecular profiling revealed an EGFR mutation. 
Patient previously received Pembrolizumab immunotherapy but showed disease progression.
"""


# 3. The Extraction Layer Simulation
# In a production app, you pass the 'PatientProfile' schema directly to models like GPT-4o
# or Claude 3.5 Sonnet, forcing them to return structured JSON matching this exact blueprint.
def simulate_llm_extraction(text: str) -> PatientProfile:
    print("Sending raw notes to Extraction Agent...")

    # This simulates the structured JSON data returned by a state-of-the-art LLM
    mock_llm_json_output = {
        "cancer_type": "Non-Small Cell Lung Cancer",
        "stage": "Stage IV",
        "mutations": ["EGFR"],
        "prior_treatments": ["Pembrolizumab"],
    }

    # Feed the JSON into our Pydantic model. It validates every data type automatically.
    validated_profile = PatientProfile(**mock_llm_json_output)
    return validated_profile


# 4. Execute the Step
if __name__ == "__main__":
    print("--- Running Rung 2: Structured Extraction ---")
    patient_data = simulate_llm_extraction(raw_doctor_notes)

    print("\n--- Extraction & Validation Successful ---")
    print(f"Validated Object:  {type(patient_data)}")
    print(f"Extracted Disease: {patient_data.cancer_type}")
    print(f"Detected Stage:    {patient_data.stage}")
    print(f"Genetic Mutations: {patient_data.mutations}")
