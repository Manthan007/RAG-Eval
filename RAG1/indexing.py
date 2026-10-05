from dotenv import load_dotenv
from file_reader import extract_paragraphs
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = InMemoryVectorStore(embedding)

chunks = extract_paragraphs("D:\\GenAI-Projects\\RAG-Eval\\pdfs\\Penguins_ACL.pdf")
vector_store.add_texts(texts=chunks)
print(f"indexed {len(chunks)} chunks")
