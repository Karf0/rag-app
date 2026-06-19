Decided stack is - FastAPI, PostgreSQL with SQLAlchemy + asyncpg, pgvector for vector storing - creating an interface that can be later easily migrated to dedicated 
vector DB (Chroma DB), local embeddings via sentence-transformers, an LLM API for generation

FastAPI 
 - for async workflow and AI oriented apps is a clear winner, already had a little bit of experience from the MINI-RAG project
 - native async 
Why not Django? 
 - not ideal for this project 
 - oriented for server-rendered pages app, batteries don't apply here
Why pgvector first instead of a Chroma DB? 
 - it simplifies the first MVP
 - create the infrastructure in a way it is easy to migrate to ChromaDB - valuable skill on its own
Local embeddings
 - don't need real big power 
 - learns more than calling an API

=========v1==============

Atomicity with pg for both of the databases is impossible if we want to migrate later. It is off the table,
we accept orphan chunks but not vectors. That means first store chunks, then vectors. For now the "orphan" chunks
is a documented-problem, which will be resolved in later versions. 

Chunk_id is code generated and not SERIAL from DB. Exists before any I/O, doesn't force the ordering (first chunk, then vector).
Vector Store holds only (chunk_id, vector) - Chroma-shaped
Retrieval in two-steps: search -> ids -> fetch text

sentence-transformer 
 - best for v1 MVP, easy, local
 - all-MiniLM-L6-v2 - fast on CPU, no GPU needed, relatively small, industry standard, not the strongest on embedding but for MVP enough, easy to swap later for better model based on MTEB
 - for similarity search we use cosine_distance - all-MiniLM-l6-v2 optimized for that

Storing just the path to a raw documnet to give the user
Later use cross-encoders for better retrieval - not in a v1

Two method in embedding class
 - many tranformers use different one for each one so for future replacement is better to use this interface
 - one embedd_document 
 - one embedd_query

Seperate chunk_store, doc_store, vector_store (swappable).
