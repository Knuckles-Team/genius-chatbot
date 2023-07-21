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
import nltk
from typing import List
from multiprocessing import Pool
from tqdm import tqdm
from chromadb.config import Settings
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import GPT4All, LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
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
nltk.download('punkt')


class MyElmLoader(UnstructuredEmailLoader):
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
        self.persist_directory = f'{os.path.normpath(os.path.dirname(__file__))}/chromadb'
        self.model_directory = f'{os.path.normpath(os.path.dirname(__file__))}/models'
        self.source_directory = os.path.normpath(os.path.dirname(__file__))
        self.model = "wizardlm-13b-v1.1-superhot-8k.ggmlv3.q4_0.bin"
        self.model_path = os.path.normpath(os.path.join(self.model_directory, self.model))
        self.model_engine = "GPT4All"
        self.embeddings_model_name = "all-MiniLM-L6-v2"
        self.set_directory(directory=os.path.normpath(os.path.dirname(__file__)))
        self.chunk_overlap = 69
        self.chunk_size = 639
        self.target_source_chunks = 6
        self.mute_stream = False
        self.hide_source = False
        self.model_n_ctx = 2127
        self.model_n_batch = 9
        self.bytes = 1073741824
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
            ".eml": (MyElmLoader, {}),
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

    def set_directory(self, directory):
        self.persist_directory = f'{directory}/chromadb'
        self.model_directory = f'{directory}/models'
        if os.path.isdir(self.model_directory):
            print("Models directory exists")
        else:
            print(f"Making models directory: {self.model_directory}")
            os.mkdir(self.model_directory)
        self.model_path = os.path.normpath(os.path.join(self.model_directory, self.model))
        self.chroma_settings = Settings(
            chroma_db_impl='duckdb+parquet',
            persist_directory=self.persist_directory,
            anonymized_telemetry=False
        )

    def check_hardware(self):
        self.total_memory = psutil.virtual_memory()[0]
        self.used_memory = psutil.virtual_memory()[3]
        self.free_memory = psutil.virtual_memory()[1]
        self.used_percent = psutil.virtual_memory()[2]
        print(f'RAM Utilization: {round(self.used_percent, 2)}%\n'
              f'\tUsed  RAM: {round(float(self.used_memory / self.bytes), 2)} GB\n'
              f'\tFree  RAM: {round(float(self.free_memory / self.bytes), 2)} GB\n'
              f'\tTotal RAM: {round(float(self.total_memory / self.bytes), 2)} GB\n\n')

    def chat(self, prompt):
        embeddings = HuggingFaceEmbeddings(model_name=self.embeddings_model_name)
        db = Chroma(persist_directory=self.persist_directory,
                    embedding_function=embeddings,
                    client_settings=self.chroma_settings)
        retriever = db.as_retriever(search_kwargs={"k": self.target_source_chunks})
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
        embeddings = HuggingFaceEmbeddings(model_name=self.embeddings_model_name)

        if self.does_vectorstore_exist():
            # Update and store locally vectorstore
            print(f"Appending to existing vectorstore at {self.persist_directory}")
            db = Chroma(persist_directory=self.persist_directory, embedding_function=embeddings,
                        client_settings=self.chroma_settings)
            collection = db.get()
            texts = self.process_documents([metadata['source'] for metadata in collection['metadatas']])
            print(f"Creating embeddings. May take some minutes...")
            db.add_documents(texts)
        else:
            # Create and store locally vectorstore
            print("Creating new vectorstore")
            texts = self.process_documents()
            print(f"Creating embeddings. May take some minutes...")
            db = Chroma.from_documents(texts, embeddings, persist_directory=self.persist_directory,
                                       client_settings=self.chroma_settings)
        db.persist()
        db = None
        print(f"Ingestion complete! You can now run genius-chatbot to query your documents")

    def load_single_document(self, file_path: str) -> List[Document]:
        ext = "." + file_path.rsplit(".", 1)[-1]
        if ext in self.loader_mapping:
            loader_class, loader_args = self.loader_mapping[ext]
            loader = loader_class(file_path, **loader_args)
            return loader.load()

        raise ValueError(f"Unsupported file extension '{ext}'")

    def load_documents(self, source_dir: str, ignored_files=None) -> List[Document]:
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

        with Pool(processes=os.cpu_count()) as pool:
            results = []
            with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
                for i, docs in enumerate(pool.imap_unordered(self.load_single_document, filtered_files)):
                    results.extend(docs)
                    pbar.update()

        return results

    def process_documents(self, ignored_files=None) -> List[Document]:
        """
        Load documents and split in chunks
        """
        if ignored_files is None:
            ignored_files = []
        print(f"Loading documents from {self.source_directory}")
        documents = self.load_documents(self.source_directory, ignored_files)
        if not documents:
            print("No new documents to load")
            return []
        print(f"Loaded {len(documents)} new documents from {self.source_directory}")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        texts = text_splitter.split_documents(documents)
        print(f"Split into {len(texts)} chunks of text (max. {self.chunk_size} tokens each)")
        return texts

    def does_vectorstore_exist(self) -> bool:
        """
        Checks if vectorstore exists
        """
        if os.path.exists(os.path.join(self.persist_directory, 'index')):
            if os.path.exists(os.path.join(self.persist_directory, 'chroma-collections.parquet')) \
                    and os.path.exists(os.path.join(self.persist_directory, 'chroma-embeddings.parquet')):
                list_index_files = glob.glob(os.path.join(self.persist_directory, 'index/*.bin'))
                list_index_files += glob.glob(os.path.join(self.persist_directory, 'index/*.pkl'))
                # At least 3 documents are needed in a working vectorstore
                if len(list_index_files) > 3:
                    return True
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
                                    'embeddings-model=', 'model=', 'model-engine='])
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            usage()
            sys.exit()
        elif opt in ('-a', '--assimilate'):
            if os.path.exists(arg):
                geniusbot_chat.source_directory = str(arg)
                assimilate_flag = True
            else:
                print(f'Path does not exist: {arg}')
                sys.exit(1)
        elif opt in ('-b', '--batch-token'):
            geniusbot_chat.model_n_batch = int(arg)
        elif opt in ('-c', '--chunks'):
            geniusbot_chat.target_source_chunks = int(arg)
        elif opt in ('-d', '--directory'):
            if os.path.exists(arg):
                geniusbot_chat.set_directory(directory=str(arg))
            else:
                print(f'Path does not exist: {arg}')
                sys.exit(1)
        elif opt in ('-j', '--json'):
            geniusbot_chat.json_export_flag = True
            geniusbot_chat.hide_source_flag = True
            geniusbot_chat.mute_stream_flag = True
        elif opt in ('-e', '--embeddings-model'):
            geniusbot_chat.embeddings_model_name = arg
        elif opt in ('-m', '--model'):
            geniusbot_chat.model = arg
            geniusbot_chat.model_path = os.path.normpath(os.path.join(geniusbot_chat.model_directory, geniusbot_chat.model))
            print(f"Model: {geniusbot_chat.model}")
        elif opt in ('-x', '--model-engine'):
            geniusbot_chat.model_engine = arg
            if geniusbot_chat.model_engine.lower() != "llamacpp" and geniusbot_chat.model_engine.lower() != "gpt4all":
                print("model type not supported")
                usage()
                sys.exit(2)
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

    response = "Empty"
    if run_flag:
        geniusbot_chat.assimilate()
        print('RAM Utilization Before Loading Model')
        geniusbot_chat.check_hardware()
        response = geniusbot_chat.chat(prompt)
        print('RAM Utilization After Loading Model')
        geniusbot_chat.check_hardware()

    if json_export_flag:
        print(json.dumps(response, indent=4))
    else:
        print(f"Question: {response['prompt']}\n"
              f"Answer: {response['answer']}\n"
              f"Sources: {response['sources']}")


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
