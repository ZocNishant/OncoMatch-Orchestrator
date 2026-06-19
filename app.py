import random
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# ====================================================
# 1. INITIALIZE WEB SERVER & LOCAL DATABASE
# ====================================================
app = FastAPI(title="OncoMatch Orchestrator API", version="1.0")

# Setup our in-memory vector database when the server boots up
db_client = QdrantClient(":memory:")
COLLECTION_NAME = "api_trials"
VECTOR_DIMENSION = 384

db_client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=VECTOR_DIMENSION, distance=Distance.COSINE),
)

# Seed our mock database with the target clinical trial
mock_protocol = "Protocol: Stage IV Non-Small Cell Lung Cancer. Exclude if patient has active brain metastases."
db_client.upsert(
    collection_name=COLLECTION_NAME,
    points=[
        PointStruct(
            id=str(uuid.uuid4()),
            vector=[
                random.uniform(-1, 1) for _ in range(VECTOR_DIMENSION)
            ],  # simple seed vector
            payload={
                "trial_id": "NCT0456123",
                "text_content": mock_protocol,
                "exclusions": ["brain metastases"],
            },
        )
    ],
)


# ====================================================
# 2. DEFINE THE INCOMING WEB REQUEST STRUCT (Pydantic)
# ====================================================
# This forces anyone sending a web request to provide data in this exact JSON format
class MatchRequest(BaseModel):
    doctor_notes: str
    has_brain_metastases: bool


# ====================================================
# 3. CREATE THE LIVE WEB ENDPOINT
# ====================================================
@app.post("/match-patient")
async def match_patient_endpoint(request: MatchRequest):
    """
    Accepts clinical patient notes over the web and runs the matching pipeline.
    """
    try:
        print(
            f"\n[API Server Received Request]: Processing incoming patient payload..."
        )

        # Simulated Agent Step 1: Extracting (We inspect what the user sent us)
        notes = request.doctor_notes
        has_metastases = request.has_brain_metastases

        # Simulated Agent Step 2: Retrieve from Vector DB
        # Creating a predictable vector for our search
        random.seed(len(notes))
        query_vector = [random.uniform(-1, 1) for _ in range(VECTOR_DIMENSION)]

        results = db_client.query_points(
            collection_name=COLLECTION_NAME, query=query_vector, limit=1
        )
        matched_trial = results.points[0].payload

        # Simulated Agent Step 3: The Critic Rule Guard
        if has_metastases and "brain metastases" in matched_trial["exclusions"]:
            status = "REJECTED"
            reason = f"Patient has active brain metastases, which explicitly violates exclusion criteria for trial {matched_trial['trial_id']}."
        else:
            status = "APPROVED"
            reason = f"Patient meets target profile metrics for Trial {matched_trial['trial_id']}."

        # Return a structured web response back to the client application
        return {
            "status": status,
            "matched_trial_id": matched_trial["trial_id"],
            "evaluation_details": reason,
            "system_processed_successfully": True,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Internal Server Error in Agent Flow: {str(e)}"
        )
