from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
import os
import shutil


# langchain_community...

Chroma_path = "chroma"
Data_path = "Data"

def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)




def load_documents():
    loader = DirectoryLoader(Data_path)
    documents = loader.load()
    return documents



def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks

def save_to_chroma(chunks: list[Document]):
    if os.path.exists(Chroma_path):
        shutil.rmtree(Chroma_path)

    # create new db 
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=Chroma_path
    )

    db.persist()
    print(f"Saved {len(chunks)} chunks to {Chroma_path}.")

if __name__ == "__main__":
    main()