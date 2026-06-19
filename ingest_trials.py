import random
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# 1. Initialize Qdrant Client in Local Memory Mode
# This allows us to prototype instantly without running an external server.
print("Initializing local Vector Database...")
client = QdrantClient(":memory:")

COLLECTION_NAME = "clinical_trials"
VECTOR_DIMENSION = 384  # Standard size for lightweight biomedical embedding models

# 2. Create the Collection (Think of this as a SQL Table, but for Vectors)
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=VECTOR_DIMENSION,
        distance=Distance.COSINE,  # Measures the angle between vectors to check similarity
    ),
)

# 3. Raw Mock Clinical Data (Simulating a PDF download from ClinicalTrials.gov)
mock_clinical_trial_pdf = """
Protocol ID: NCT0456123
Title: Targeted Therapy Study for Advanced Non-Small Cell Lung Cancer (NSCLC)
Inclusion Criteria: Patients must have Stage IV Non-Small Cell Lung Cancer. Must test positive for the EGFR genetic mutation. Age greater than 18.
Exclusion Criteria: Prior treatment with tyrosine kinase inhibitors (TKIs). Active brain metastases. Pregnant or lactating individuals.
"""


# 4. The Chunking Component
# In production, we break massive PDFs down so the LLM doesn't get overwhelmed.
def chunk_text(text: str, chunk_size: int = 150) -> list:
    words = text.split()
    return [
        " ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)
    ]


text_chunks = chunk_text(mock_clinical_trial_pdf)


# 5. The Embedding Generator Simulation
# Real systems pass text to an embedding model (like PubMedBERT) which returns floats.
# We will simulate this by generating an array of random floats for our dimension size.
def generate_mock_embedding(text: str, dimension: int = 384) -> list:
    # seeding to simulate semi-deterministic vector generation based on contents
    random.seed(len(text))
    return [random.uniform(-1, 1) for _ in range(dimension)]


# 6. Structuring Data into Qdrant "Points"
points_to_upload = []
for i, chunk in enumerate(text_chunks):
    vector = generate_mock_embedding(chunk, dimension=VECTOR_DIMENSION)

    # Each item in a Vector DB is called a 'Point'. It requires an ID, a Vector, and a Payload (Metadata)
    point = PointStruct(
        id=str(uuid.uuid4()),  # Generates a unique string ID
        vector=vector,
        payload={
            "trial_id": "NCT0456123",
            "cancer_type": "Lung Cancer",
            "chunk_index": i,
            "text_content": chunk,
        },
    )
    points_to_upload.append(point)

# 7. Upsert (Insert/Update) into our Vector Database
client.upsert(collection_name=COLLECTION_NAME, points=points_to_upload)
print(
    f"Successfully chunked, embedded, and indexed {len(points_to_upload)} points into Qdrant."
)

# 8. Test Query: Let's simulate searching the database
print("\n--- Testing Vector Search Retrieval ---")
search_query = "Looking for Stage IV lung cancer trials with mutations"
query_vector = generate_mock_embedding(search_query, dimension=VECTOR_DIMENSION)

# FIX: Modern Qdrant uses query_points() instead of search()
search_results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,  # Passed directly to the 'query' parameter
    limit=1,  # Give us top 1 most similar match
)

# FIX: Iterate through search_results.points instead of the raw result object
for hit in search_results.points:
    print(f"Match Found (Confidence Score: {hit.score:.4f}):")
    print(f"Trial ID: {hit.payload['trial_id']}")
    print(f"Content snippet: {hit.payload['text_content'][:120]}...")
