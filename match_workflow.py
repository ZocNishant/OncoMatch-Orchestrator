import random
import uuid

from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# ==========================================
# 1. SETUP FOUNDATION & MODELS (From Rungs 1 & 2)
# ==========================================
client = QdrantClient(":memory:")
COLLECTION_NAME = "clinical_trials"
VECTOR_DIMENSION = 384

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=VECTOR_DIMENSION, distance=Distance.COSINE),
)


class PatientProfile(BaseModel):
    cancer_type: str
    stage: str
    mutations: list
    prior_treatments: list


def generate_mock_embedding(text: str) -> list:
    random.seed(len(text))
    return [random.uniform(-1, 1) for _ in range(VECTOR_DIMENSION)]


# Seed the database with our specific trial protocol
mock_trial_text = """
Protocol ID: NCT0456123
Target: Non-Small Cell Lung Cancer (NSCLC).
Inclusion: Must be Stage IV and EGFR mutation positive.
Exclusion: Disqualified if patient has active brain metastases.
"""
client.upsert(
    collection_name=COLLECTION_NAME,
    points=[
        PointStruct(
            id=str(uuid.uuid4()),
            vector=generate_mock_embedding(mock_trial_text),
            payload={
                "trial_id": "NCT0456123",
                "text_content": mock_trial_text,
                "exclusion_keywords": ["brain metastases"],
            },
        )
    ],
)

# ==========================================
# 2. DEFINING THE ORCHESTRATION PIPELINE
# ==========================================


# Rung 2 Extraction Agent
def extraction_agent(raw_notes: str) -> PatientProfile:
    print("\n[Extraction Agent]: Parsing messy doctor notes...")
    # Simulating structural parsing from raw text
    return PatientProfile(
        cancer_type="Non-Small Cell Lung Cancer",
        stage="Stage IV",
        mutations=["EGFR"],
        prior_treatments=["Pembrolizumab"],
    )


# Retriever Agent
def retriever_agent(profile: PatientProfile) -> dict:
    print(
        f"\n[Retriever Agent]: Querying Vector DB for '{profile.cancer_type}' trials..."
    )
    query_str = f"Trials for {profile.cancer_type} {profile.stage} with {', '.join(profile.mutations)} mutations"
    query_vec = generate_mock_embedding(query_str)

    results = client.query_points(
        collection_name=COLLECTION_NAME, query=query_vec, limit=1
    )
    matched_trial = results.points[0].payload
    return matched_trial


# Critic Agent (The Gatekeeper)
def critic_agent(
    profile: PatientProfile, trial_payload: dict, patient_has_metastases: bool
) -> str:
    print("\n[Critic Agent]: Verifying eligibility and parsing exclusion criteria...")

    # Check if the patient has a condition that explicitly triggers the trial's exclusions
    exclusions = trial_payload["exclusion_keywords"]

    if patient_has_metastases and "brain metastases" in exclusions:
        return f"REJECTED: Patient has active brain metastases, which violates the exclusion rules for trial {trial_payload['trial_id']}."
    else:
        return f"APPROVED: Patient is a strong candidate for Trial {trial_payload['trial_id']}."


# ==========================================
# 3. RUNNING THE ORCHESTRATOR LOOP
# ==========================================
if __name__ == "__main__":
    print("--- Starting Rung 3: Stateful Workflow Pipeline ---")

    # Case Scenario: Patient notes indicate advanced stage, but we also discover they have brain metastases
    doctor_notes = "Patient has Stage IV NSCLC with EGFR mutation. Brain MRI indicates active metastases."
    has_brain_metastases = True  # Extracted diagnostic flag

    # Step 1: Extract
    patient_profile = extraction_agent(doctor_notes)

    # Step 2: Retrieve
    matched_trial_data = retriever_agent(patient_profile)
    print(f"-> Found Trial Match: {matched_trial_data['trial_id']}")

    # Step 3: Critically Evaluate
    final_decision = critic_agent(
        patient_profile, matched_trial_data, has_brain_metastases
    )

    print("\n================ FINAL SYSTEM OUTCOME ================")
    print(final_decision)
    print("======================================================")
