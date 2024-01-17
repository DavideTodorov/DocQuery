from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader, TextLoader
from chainlit.types import AskFileResponse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
import chainlit as cl

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
embeddings = OpenAIEmbeddings()


def process_file(file: AskFileResponse):
    import tempfile

    if file.type == "text/plain":
        loader = TextLoader
    elif file.type == "application/pdf":
        loader = PyPDFLoader

    with tempfile.NamedTemporaryFile() as tempfile:
        tempfile.write(file.content)
        loader = loader(tempfile.name)
        documents = loader.load()
        documents_split = text_splitter.split_documents(documents)

        for i, doc in enumerate(documents_split):
            doc.metadata["source"] = f"source_{i}"

        return documents_split


def get_docsearch(file: AskFileResponse):
    docs = process_file(file)

    cl.user_session.set("docs", docs)

    return Chroma.from_documents(docs, embeddings)


welcome_message = """Welcome to DocQuery! To get started:
1. Upload a PDF or text file
2. Ask a question about the file
"""


async def get_file_from_user():
    await cl.Message(content="Hello! Please upload a pdf in order to ask questions about it.").send()

    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content=welcome_message,
            accept=["text/plain", "application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    return files[0]
