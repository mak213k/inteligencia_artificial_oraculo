#pip install pypdf
#pip install python-dotenv
#pip install langchain-core
#pip install langchain
#pip install loguru
#pip install langchain_huggingface
#pip install langchain_community
# pip install faiss-gpu-cu12[fix-cuda]
#pip install langchain


import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from dotenv import load_dotenv
from langchain import hub
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory
#from langchain.retrievers import ScoreThresholdRetriever
from langchain.chains import (
    StuffDocumentsChain, LLMChain, ConversationalRetrievalChain
)

from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

from langchain_community.vectorstores.faiss import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import HuggingFacePipeline
from langchain_community.llms import LlamaCpp

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.callbacks import FileCallbackHandler, StdOutCallbackHandler
from langchain_core.prompts import BasePromptTemplate
from langgraph.checkpoint.memory import MemorySaver

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.documents import Document

from huggingface_hub import create_repo
# import { ScoreThresholdRetriever } from "langchain/retrievers/score_threshold";

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

from typing_extensions import List, TypedDict
from langgraph.graph import START, StateGraph
from loguru import logger

# Configuração inicial
load_dotenv(".env")

# Diretório padrão dos PDFs
DIRETORIO_ARQUIVO = os.getenv("DIRETORIO_ARQUIVO")
MODEL_PATH = "../../model/"  # Caminho do modelo LLaMA # Caminho do modelo LLaMA
INDEX_PATH = "./indice_faiss"  # Local para persistir o índice FAISS


os.environ['CURL_CA_BUNDLE'] = '/home/suporte/ambiente_ia/oraculo_v1/certificado/ProxyCA.cer'
os.environ['REQUESTS_CA_BUNDLE'] = '/home/suporte/ambiente_ia/oraculo_v1/certificado/ProxyCA.cer'


class Oraculo:
    #def __init__(self):
    # 1. Carregar documentos em PDFs


    # Define state for application
    class State(TypedDict):
        question: str
        # query: Search
        context: List[Document]
        answer: str


    # def format_docs(self):
    #     return "\n\n".join(doc.page_content for doc in state['docs'])


    # def analyze_query(self):
    #     structured_llm = llm.with_structured_output(Search)
    #     query = structured_llm.invoke(state["question"])
    #     return {"query": query}


    def carregar_documentos(self,pasta):
        documentos = []
        for arquivo in os.listdir(pasta):
            if arquivo.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(pasta, arquivo))
                documentos.extend(loader.load_and_split())
        return documentos


    # 2. Inicializar embeddings (SentenceTransformers)
    def carregar_embeddings(self):
        model_name = MODEL_PATH+'all-mpnet-base-v2'
        # model_kwargs = {'device': 'cpu'}
        model_kwargs = {
            'device': 'cuda',
        }
        encode_kwargs = {'normalize_embeddings': False}
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

        return embeddings  # Modelo leve e eficiente


    # 3. Criar ou carregar índice FAISS
    def criar_ou_carregar_indice(self, documentos,embeddings,allow_dangerous_deserialization=True):
        if os.path.exists(INDEX_PATH+"/index.faiss"):
            print("Carregando índice FAISS existente...")
            return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

        print("Criando novo índice FAISS...")
        vectorstore = FAISS.similarity_search_with_score(documentos, embeddings)
        vectorstore.save_local(INDEX_PATH)
        return {"vectorstore": vectorstore}


    # 4. Inicializar o modelo LLaMA
    def carregar_modelo_llama(self):
        print("Carregando modelo LLaMA...")

        model_kwargs={
            'gpu_layers_only': True,
            'multi_process' : True,
            'show_progress' : True,
            'device' : "cuda"  # Executar na GPU
        }

        llm_model = LlamaCpp(
            #model_path=MODEL_PATH+"llama32_1b.gguf",
            model_path=MODEL_PATH + "llama-3.2-3b-instruct-q8_0.gguf",
            temperature=0.1,
            n_ctx=13000,
            n_gpu_layers=5000,  # Camadas processadas na GPU
            use_mmap=False,
            use_mlock=True,
            model_kwargs=model_kwargs,
            #device="cpu"
        )

        # Envolver o modelo no wrapper
        return llm_model

    def configurar_runnable(self, llm, retriever):
        # 2. Incorporate the retriever into a question-answering chain.
        system_prompt = (
            "You are a task assistant of answering questions."
            "Write a concise summary of the following: {context} ."
            "Use the parts of the retrieved context to answer."
            "the question. If you don't know the answer, say: "
            "Desculpe. Não achei este conteúdo. A pesquisa pode ser feita no portal: http://portais.guarulhos.sp.gov.br:8080/apex/f?p=111:10:8442991627610405::::: ."
            "Use a maximum of three sentences and keep the answer concise."
            "Answer strictly what the user asked."
            "Always answer in Brazilian Portuguese."
            "Only bring document high adherent at the input."
        )

        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
        history_aware_retriever = create_history_aware_retriever(llm, retriever, prompt_template)

        combine_docs_chain = create_stuff_documents_chain(llm, prompt_template)

        # combine_docs_chain = create_stuff_documents_chain(
        #     llm, retrieval_qa_chat_prompt
        # )

        # print(question_answer_chain)
        # exit(0)

        # ("context", retriever | format_docs),

        chain = create_retrieval_chain(history_aware_retriever, combine_docs_chain)

        return chain



    # 6. Perguntar ao modelo
    def perguntar(self, qa_chain, chat_history, pergunta):

        logfile = "output.log"
        logger.add(logfile, colorize=True, enqueue=True)
        # handler_1 = FileCallbackHandler(logfile)
        # handler_2 = StdOutCallbackHandler()
        # , {"callbacks": [handler_1, handler_2]}

        resultado = qa_chain.invoke({"input": pergunta, "chat_history": chat_history})
        # resultado = retrieve_with_threshold(pergunta, qa_chain)

        return resultado

        # print(f"\nResposta: {resultado['context']}\n")
        # print("Documentos Relevantes:")
        # for doc in resultado:
        #     print(f"\n\n")
        #     print(f" lei e número: {doc.metadata}")
        #     print(f"\n\n")
        #     print(f" conteúdo: {doc.page_content}")
        #     print(doc)
        #     print(" fim... ")
        #     exit(0)

    def agregadorRetriever(self, pergunta):
        print("Inicializando sistema...")

        # Carregar documentos e embeddings
        documentos = self.carregar_documentos(DIRETORIO_ARQUIVO)
        embeddings = self.carregar_embeddings()

        # Criar ou carregar índice FAISS
        # vectorstore = criar_ou_carregar_indice(documentos, embeddings)

        # Configurar o modelo LLaMA e o QA Chain.
        # Optei por utilizar o método similarity_search_with_score que me trás melhor resultado.
        # retriever = vectorstore.as_retriever(
        #     search_type="similarity",  # Muda o tipo de busca para "similarity".
        #      search_kwargs={
        #          "k": 5,  # Define o número máximo de documentos a retornar.
        #          "mmr": True,  # Garante diversidade nos documentos retornados.
        #          "lambda": 0.9,  # Ajusta o balanceamento entre relevância (90%) e diversidade (10%).
        #          "score_threshold": 0.5,
        #      },
        # )

        db = FAISS.from_documents(documentos, embeddings)
        results_with_scores = db.similarity_search_with_score(pergunta)

        return results_with_scores,db


    def agregadorChain(self, llm, vectorstore, pergunta):
        print("Inicializando sistema...")

        # # Carregar documentos e embeddings
        # documentos = carregar_documentos(DIRETORIO_ARQUIVO)
        embeddings = self.carregar_embeddings()


        # Configurar o modelo LLaMA e o QA Chain
        # retriever = vectorstore.as_retriever(
        #     search_type="mmr",  # Muda o tipo de busca para "similarity".
        #     search_kwargs={
        #         "k": 5,  # Define o número máximo de documentos a retornar.
        #         "mmr": True,  # Garante diversidade nos documentos retornados.
        #         "lambda": 0.7,  # Ajusta o balanceamento entre relevância (90%) e diversidade (10%).
        #     },
        # )

        qa_chain = self.configurar_runnable(llm, vectorstore.as_retriever())
        return qa_chain

    # Filtro personalizado para aplicar score_threshold após a busca
    def retrieve_with_threshold(self,query, retriever, threshold=0.2):
        results = retriever.invoke(query)
        filtered_results = [doc for doc in results if doc.metadata.get("score", 0) >= threshold]
        return filtered_results



    def responder(self,chat_history, pergunta):
        # # Criar ou carregar índice FAISS
        # vectorstore = criar_ou_carregar_indice(documentos, embeddings)
        results_with_scores, vectorstore = self.agregadorRetriever(pergunta)
        llm = self.carregar_modelo_llama()


        chain = self.agregadorChain(llm, vectorstore, pergunta)
        return self.perguntar(chain, chat_history , pergunta)


    def conversa(self,pergunta):
        chat_history = []

        resposta = self.responder(chat_history, pergunta)

        chat_history.extend(
            [
                HumanMessage(content=pergunta),
                AIMessage(content=resposta["answer"]),
            ]
        )

        return resposta



# Execução principal
if __name__ == "__main__":
    while True:
        pergunta = input("Digite sua pergunta (ou 'sair' para encerrar): ")
        if pergunta.lower() == "sair":
            break

        oraculo = Oraculo()
        resposta = oraculo.conversa(pergunta)


        print(f" resposta: {resposta['answer']}")
        # print(f" \ntotal: {resposta}")


