from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transcript_loader import chunk
from dotenv import load_dotenv
load_dotenv()

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

vector_store = Chroma.from_documents(
    chunk,
    embedding_model
)

retriever = vector_store.as_retriever(
    search_type='mmr',
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.7
    }
)
