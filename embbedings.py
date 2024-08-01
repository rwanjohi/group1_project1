import os
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from credentials import * 

# load models
os.environ["OPENAI_API_KEY"] = API_KEY
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")
llm=OpenAI()

# Raw data: load Baba's pdf document
from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("baba_citation.pdf")
docs = loader.load()

# Split text into appropriate lengths
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)

# Compute embeddings and store in Vector database
db = Chroma.from_documents(
    documents= texts,
    embedding= embeddings_model
)

# Create a qa chain that will retrieve most relevant chunk and feed to llm
chain = RetrievalQA.from_chain_type(llm=llm,
                                    chain_type="stuff",
                                    retriever=db.as_retriever())

query = "Highlight Baba's life journey and achievements"
result = chain(query)