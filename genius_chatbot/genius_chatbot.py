#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gpt4all
import psutil
import json
import time
import sys
import getopt
import os
import glob
import hashlib
import chromadb
import logging
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import List
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import GPT4All, LlamaCpp
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

from langchain.docstore.document import Document
from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)


class MyEmlLoader(UnstructuredEmailLoader):
    """Wrapper to fallback to text/plain when default does not work"""

    def load(self) -> List[Document]:
        """Wrapper adding fallback for elm without html"""
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if 'text/html content not found in email' in str(e):
                    # Try plain text
                    self.unstructured_kwargs["content_source"] = "text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            # Add file_path to exception message
            raise type(e)(f"{self.file_path}: {e}") from e

        return doc


class ChatBot:
    def __init__(self):
        self.total_memory = psutil.virtual_memory()[0]
        self.used_memory = psutil.virtual_memory()[3]
        self.free_memory = psutil.virtual_memory()[1]
        self.used_percent = psutil.virtual_memory()[2]
        self.script_path = os.path.normpath(os.path.dirname(__file__))
        self.persist_directory = (f'{os.path.normpath(os.path.dirname(self.script_path.rstrip("/")))}'
                                  f'/chromadb')
        self.chromadb_client = chromadb.PersistentClient(path=self.persist_directory)
        self.model_directory = (f'{os.path.normpath(os.path.dirname(self.script_path.rstrip("/")))}'
                                f'/models')
        self.source_directory = os.path.normpath(os.path.dirname(__file__))
        self.model = "wizardlm-13b-v1.1-superhot-8k.ggmlv3.q4_0.bin"
        self.model_path = os.path.normpath(os.path.join(self.model_directory, self.model))
        self.model_engine = "GPT4All"
        self.embeddings_model_name = "all-MiniLM-L6-v2"
        self.embeddings = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embeddings_model_name
        )
        self.chunk_overlap = 69
        self.chunk_size = 639
        self.target_source_chunks = 6
        self.mute_stream = False
        self.hide_source = False
        self.model_n_ctx = 2127
        self.model_n_batch = 9
        self.bytes = 1073741824
        self.collection = None
        self.collection_name = "genius"
        self.chroma_settings = Settings(
            chroma_db_impl='duckdb+parquet',
            persist_directory=self.persist_directory,
            anonymized_telemetry=False
        )
        self.loader_mapping = {
            ".csv": (CSVLoader, {}),
            ".doc": (UnstructuredWordDocumentLoader, {}),
            ".docx": (UnstructuredWordDocumentLoader, {}),
            ".enex": (EverNoteLoader, {}),
            ".eml": (MyEmlLoader, {}),
            ".epub": (UnstructuredEPubLoader, {}),
            ".html": (UnstructuredHTMLLoader, {}),
            ".md": (UnstructuredMarkdownLoader, {}),
            ".odt": (UnstructuredODTLoader, {}),
            ".pdf": (PyMuPDFLoader, {}),
            ".ppt": (UnstructuredPowerPointLoader, {}),
            ".pptx": (UnstructuredPowerPointLoader, {}),
            ".txt": (TextLoader, {"encoding": "utf8"}),
            # Add more mappings for other file extensions and loaders as needed
        }
        self.payload = None

    def set_chromadb_directory(self, directory):
        self.persist_directory = f'{directory}/chromadb'
        if not os.path.isdir(self.persist_directory):
            logging.info(f"Making chromadb directory: {self.persist_directory}")
            os.mkdir(self.persist_directory)
        self.chroma_settings = Settings(
            chroma_db_impl='duckdb+parquet',
            persist_directory=self.persist_directory,
            anonymized_telemetry=False
        )
        self.chromadb_client = chromadb.PersistentClient(path=self.persist_directory)

    def set_models_directory(self, directory):
        self.model_directory = f'{directory}/models'
        if not os.path.isdir(self.model_directory):
            logging.info(f"Making models directory: {self.model_directory}")
            os.mkdir(self.model_directory)
        self.model_path = os.path.normpath(os.path.join(self.model_directory, self.model))

    def check_hardware(self):
        self.total_memory = psutil.virtual_memory()[0]
        self.used_memory = psutil.virtual_memory()[3]
        self.free_memory = psutil.virtual_memory()[1]
        self.used_percent = psutil.virtual_memory()[2]
        logging.info(f'RAM Utilization: {round(self.used_percent, 2)}%\n'
                     f'\tUsed  RAM: {round(float(self.used_memory / self.bytes), 2)} GB\n'
                     f'\tFree  RAM: {round(float(self.free_memory / self.bytes), 2)} GB\n'
                     f'\tTotal RAM: {round(float(self.total_memory / self.bytes), 2)} GB\n\n')

    def chat(self, prompt):
        llm = None
        self.collection = self.chromadb_client.get_or_create_collection(name=self.collection_name,
                                                                        embedding_function=self.embeddings)
        retriever = self.collection.as_retriever(search_kwargs={"k": self.target_source_chunks})
        # activate/deactivate the streaming StdOut callback for LLMs
        callbacks = [] if self.mute_stream else [StreamingStdOutCallbackHandler()]
        # Download model
        if os.path.isfile(os.path.join(self.model_directory, self.model)):
            print(f'Already downloaded model: {self.model_path}')
        else:
            print(f'Model was not found, downloading...')
            gpt4all.GPT4All.download_model(self.model, self.model_directory)
        # Prepare the LLM
        match self.model_engine.lower():
            case "llamaccp":
                llm = LlamaCpp(model=self.model_path, max_tokens=self.model_n_ctx, n_batch=self.model_n_batch,
                               callbacks=callbacks, verbose=False)
            case "gpt4all":
                llm = GPT4All(model=self.model_path, max_tokens=self.model_n_ctx, backend='gptj',
                              n_batch=self.model_n_batch, callbacks=callbacks, verbose=True)
            case "openai":
                pass
            case _default:
                # raise exception if model_type is not supported
                raise Exception(f"Model type {self.model_engine} is not supported. "
                                f"Please choose one of the following: LlamaCpp, GPT4All")
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever,
                                         return_source_documents=not self.hide_source)
        start = time.time()
        res = qa(prompt)
        answer, docs = res['result'], [] if self.hide_source else res['source_documents']
        end = time.time()
        documents = ""
        for document in docs:
            documents = f'{documents}\n{document.page_content}\n{document.metadata["source"]}'
        self.payload = {
            'model': self.model_engine,
            'embeddings_model': self.embeddings_model_name,
            'prompt': prompt,
            'answer': answer,
            'start_time': start,
            'end_time': end,
            'time_to_respond': round(end - start, 2),
            'batch_token': self.model_n_batch,
            'max_token_limit': self.model_n_ctx,
            'chunks': self.target_source_chunks,
            'sources': documents
        }
        return self.payload

    def assimilate(self):
        # Create embeddings
        if self.does_vectorstore_exist():
            self.collection = self.chromadb_client.get_or_create_collection(name="genius",
                                                                            embedding_function=self.embeddings)
            ids, documents, metadatas = self.process_documents()
            if documents:
                self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
                print(f"Ingestion complete! You can now run genius-chatbot to query your documents")
            else:
                print("Nothing to assimilate!")
        else:
            # Create and store locally vectorstore
            print("Creating new vectorstore")
            ids, documents, metadatas = self.process_documents()
            print(f"Creating embeddings. May take a few minutes...")
            self.collection = self.chromadb_client.get_or_create_collection(name="genius",
                                                                            embedding_function=self.embeddings)
            self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            print(f"Ingestion complete! You can now run genius-chatbot to query your documents")

    def load_single_document(self, file_path: str) -> List[Document]:
        ext = "." + file_path.rsplit(".", 1)[-1]
        if ext in self.loader_mapping:
            loader_class, loader_args = self.loader_mapping[ext]
            loader = loader_class(file_path, **loader_args)
            return loader.load()

        raise ValueError(f"Unsupported file extension '{ext}'")

    def load_documents(self, source_dir: str, ignored_files=None):
        """
        Loads all documents from the source documents directory, ignoring specified files
        """
        if ignored_files is None:
            ignored_files = []
        all_files = []
        for ext in self.loader_mapping:
            all_files.extend(
                glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
            )
        filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]
        documents = []
        id_md5 = []
        metadatas = []

        for file in filtered_files:
            ext = "." + file.rsplit(".", 1)[-1]
            if ext in self.loader_mapping:
                loader_class, loader_args = self.loader_mapping[ext]
                loader = loader_class(file, **loader_args)
                processed_docs = loader.load()
                md5_checksum = self.generate_md5_checksum(file=file)
                print(
                    f"Checking if document was already found in collection {len(self.collection.query(query_texts=[md5_checksum], n_results=1)['ids'])}")
                if len(self.collection.query(query_texts=[file], n_results=1)['ids']) > 0:
                    print(f"Document with same file name already exists, checking MD5 checksum for {file}...")
                    if len(self.collection.query(query_texts=[md5_checksum], n_results=1)['ids']) > 0:
                        print(f"Document with matching MD5 Checksum found and filename found, skipping...")
                else:
                    documents.append(processed_docs[0].page_content)
                    id_md5.append(md5_checksum)
                    metadatas.append({"source": file, "date": time.time(), "md5": md5_checksum})
        return id_md5, documents, metadatas

    def generate_md5_checksum(self, file):
        with open(file, 'rb') as file_to_check:
            # read contents of the file
            data = file_to_check.read()
            # pipe contents of the file through
            md5_checksum = hashlib.md5(data).hexdigest()
        return md5_checksum

    def process_documents(self):
        """
        Get Collection
        Compare Loaded Documents with Collection
        Load documents.
        """
        ids = []
        documents = []
        metadatas = []

        print(f"Loading documents from {self.source_directory}")
        ids, documents, metadatas = self.load_documents(self.source_directory)
        if not documents:
            print("No new documents found")
            return ids, documents, metadatas
        print(f"Loaded {len(documents)} new documents from {self.source_directory}")
        return ids, documents, metadatas

    def does_vectorstore_exist(self) -> bool:
        """
        Checks if vectorstore exists
        """
        if os.path.isfile(os.path.join(self.persist_directory, 'chroma.sqlite3')):
            return True
        else:
            return False


def usage():
    print(f'Usage:\n'
          f'-h | --help             [ See usage for script ]\n'
          f'-a | --assimilate       [ Assimilate knowledge from media provided in directory ]\n'
          f'-b | --batch-token      [ Number of tokens per batch ]\n'
          f'-c | --chunks           [ Number of chunks to use ]\n'
          f'-d | --directory        [ Directory for chromadb and model storage ]\n'
          f'-e | --embeddings-model [ Embeddings model to use https://www.sbert.net/docs/pretrained_models.html ]\n'
          f'-j | --json             [ Export to JSON ]\n'
          f'-p | --prompt           [ Prompt for chatbot ]\n'
          f'-q | --mute-stream      [ Mute stream of generation ]\n'
          f'-m | --model            [ Model to use from GPT4All https://gpt4all.io/index.html ]\n'
          f'-s | --hide-source      [ Hide source of answer ]\n'
          f'-t | --max-token-limit  [ Maximum token to generate ]\n'
          f'-x | --model-engine     [ GPT4All or LlamaCPP ]\n'
          f'\nExample:\n'
          f'genius-chatbot\n'
          f'\t--assimilate "/directory/of/documents"\n'
          f'\n'
          f'genius-chatbot\n'
          f'\t--prompt "What is the 10th digit of Pi?"\n'
          f'\n'
          f'genius-chatbot\n'
          f'\t--model "wizardlm-13b-v1.1-superhot-8k.ggmlv3.q4_0.bin"\n'
          f'\t--prompt "Chatbots are cool because they"\n'
          f'\t--model-engine "GPT4All"\n'
          f'\t--assimilate "/directory/of/documents"\n'
          f'\t--json\n')


def genius_chatbot(argv):
    geniusbot_chat = ChatBot()
    run_flag = False
    assimilate_flag = False
    json_export_flag = False
    prompt = 'Geniusbot is the smartest chatbot in existence.'
    try:
        opts, args = getopt.getopt(argv, 'a:b:c:d:e:hjm:p:q:st:x:',
                                   ['help', 'assimilate=', 'batch-token=', 'chunks=', 'directory=',
                                    'hide-source', 'mute-stream', 'json', 'prompt=', 'max-token-limit=',
                                    'embeddings-model=', 'model=', 'model-engine=', 'model-directory='])
    except getopt.GetoptError as e:
        usage()
        logging.error("Error: {e}\nExiting...")
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            usage()
            logging.error("Exiting...")
            sys.exit()
        elif opt in ('-a', '--assimilate'):
            if os.path.exists(arg):
                geniusbot_chat.source_directory = str(arg)
                assimilate_flag = True
            else:
                logging.error(f'Path does not exist: {arg}')
                sys.exit(1)
        elif opt in ('-b', '--batch-token'):
            geniusbot_chat.model_n_batch = int(arg)
        elif opt in ('-c', '--chunks'):
            geniusbot_chat.target_source_chunks = int(arg)
        elif opt in ('-d', '--directory'):
            geniusbot_chat.set_chromadb_directory(directory=str(arg))
        elif opt in ('-j', '--json'):
            geniusbot_chat.json_export_flag = True
            geniusbot_chat.hide_source_flag = True
            geniusbot_chat.mute_stream_flag = True
        elif opt in ('-e', '--embeddings-model'):
            geniusbot_chat.embeddings_model_name = arg
            geniusbot_chat.embeddings = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=geniusbot_chat.embeddings_model_name
            )
        elif opt in ('-m', '--model'):
            geniusbot_chat.model = arg
            geniusbot_chat.model_path = os.path.normpath(
                os.path.join(geniusbot_chat.model_directory, geniusbot_chat.model))
            print(f"Model: {geniusbot_chat.model}")
        elif opt in ('-x', '--model-engine'):
            geniusbot_chat.model_engine = arg
            if geniusbot_chat.model_engine.lower() != "llamacpp" and geniusbot_chat.model_engine.lower() != "gpt4all":
                logging.error("model type not supported")
                usage()
                sys.exit(2)
        elif opt == '--model-directory':
            geniusbot_chat.set_models_directory(directory=str(arg))
        elif opt in ('-p', '--prompt'):
            prompt = str(arg)
            run_flag = True
        elif opt in ('-s', '--hide-source'):
            geniusbot_chat.hide_source_flag = True
        elif opt in ('-t', '--max-token-limit'):
            geniusbot_chat.model_n_ctx = int(arg)
        elif opt in ('-q', '--mute-stream'):
            geniusbot_chat.mute_stream_flag = True

    if assimilate_flag:
        geniusbot_chat.assimilate()

    if run_flag:
        geniusbot_chat.assimilate()
        logging.info('RAM Utilization Before Loading Model')
        geniusbot_chat.check_hardware()
        response = geniusbot_chat.chat(prompt)
        if json_export_flag:
            print(json.dumps(response, indent=4))
        else:
            print(f"Question: {response['prompt']}\n"
                  f"Answer: {response['answer']}\n"
                  f"Sources: {response['sources']}")
            logging.info('RAM Utilization After Loading Model')
        geniusbot_chat.check_hardware()


def main():
    if len(sys.argv) < 2:
        usage()
        sys.exit(2)
    genius_chatbot(sys.argv[1:])


if __name__ == '__main__':
    if len(sys.argv) < 2:
        usage()
        sys.exit(2)
    genius_chatbot(sys.argv[1:])
