import os
from dotenv import load_dotenv
import gradio as gr

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
pdf_path = os.getenv("PDF_PATH")

if not api_key:
    raise ValueError("OPENAI_API_KEY is missing in .env file")

if not pdf_path:
    raise ValueError("PDF_PATH is missing in .env file")

loader = PyPDFLoader(pdf_path)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(api_key=api_key)

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)

retriever = vector_db.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    api_key=api_key
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

def answer_question(question):
    if question.strip() == "":
        return "Please enter a question."

    response = qa_chain.invoke({"query": question})
    return response["result"]

app = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(
        lines=3,
        placeholder="Ask a question about Nestlé HR policy..."
    ),
    outputs="text",
    title="Nestlé HR Policy Assistant",
    description="Ask questions based on Nestlé's HR policy document."
)
app = gr.Interface(...)
app.launch()