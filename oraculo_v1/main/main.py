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
from langchain.text_splitter import RecursiveCharacterTextSplitter
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
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.callbacks import FileCallbackHandler, StdOutCallbackHandler
from langchain_core.prompts import BasePromptTemplate
from langchain_core.retrievers import BaseRetriever

from langgraph.checkpoint.memory import MemorySaver

from langchain.schema import Document


from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from sentence_transformers import SentenceTransformer

from langchain_community.retrievers import BM25Retriever

from typing import Union
from typing import List

from fastapi import FastAPI
from fastapi import Query

import psycopg2

import unittest

from huggingface_hub import create_repo
# import { ScoreThresholdRetriever } from "langchain/retrievers/score_threshold";

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

from typing_extensions import List, TypedDict
from langgraph.graph import START, StateGraph
from loguru import logger



# Configuração inicial
load_dotenv(".env")

# Diretório padrão dos PDFs
MODEL_PATH = os.getenv("MODEL_PATH")
INDEX_PATH = os.getenv("INDEX_PATH")
EMBED_PATH = os.getenv("EMBED_PATH")
DIRETORIO_ARQUIVO = os.getenv("DIRETORIO_ARQUIVO")

# os.environ['CURL_CA_BUNDLE'] = '/home/suporte/ambiente_ia/oraculo_v1/certificado/ProxyCA.cer'
# os.environ['REQUESTS_CA_BUNDLE'] = '/home/suporte/ambiente_ia/oraculo_v1/certificado/ProxyCA.cer'

# app = FastAPI()
logfile = "output.log"
logger.add(logfile, colorize=True, enqueue=True)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,        # quantidade de tokens desejada por chunk
    chunk_overlap=50,      # sobreposição entre chunks (em tokens)
)

llm_model = None
embeddings = None


class PostgresRetriever(BaseRetriever):
    # Declara o campo para o Pydantic
    documentos: List[Document]

    def __init__(self, documentos: List[Document]):
        super().__init__(documentos=documentos)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        # Apenas retorna os documentos já filtrados no PostgreSQL
        return self.documentos

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)


# @app.on_event("startup")
def load_model():
    global llm_model
    llm_model = LlamaCpp(
        model_path=MODEL_PATH + "llama-3.2-3b-instruct-q8_0.gguf",
        temperature=0.1,
        n_batch = 64,
        # n_ctx=8192,
        n_ctx=131072,
        stop=["Q"],
        n_gpu_layers=0,
        use_mmap=True,
        use_mlock=False
    )
    print("✅ Modelo carregado na memória e pronto para uso.")

    global embeddings
    # model_name = EMBED_PATH+'gpt2-small-portuguese'
    #model_name = EMBED_PATH+'pierreguillou/bert-base-cased-squad-v1.1-portuguese'
    model_name = EMBED_PATH+'PORTULAN/serafim-335m-portuguese-pt-sentence-encoder'
    model_kwargs = {
        'device': 'cpu',
    }
    encode_kwargs = {'normalize_embeddings': False}
    
    embeddings = SentenceTransformer(model_name)

    print("✅ Embbeding carregado na memória e pronto para uso.")

load_model()


class Oraculo:
    #def __init__(self):
    # 1. Carregar documentos em PDFs


    # Define state for application
    class State(TypedDict):
        question: str
        # query: Search
        context: List[Document]
        answer: str


    def carregar_documentos(self,pasta):
        documentos = []
        for arquivo in os.listdir(pasta):
            if arquivo.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(pasta, arquivo))
                documentos.extend(loader.load_and_split(text_splitter=text_splitter))
        return documentos


    # 2. Inicializar embeddings (SentenceTransformers)
    def carregar_embeddings(self):
        model_name = EMBED_PATH+'all-mpnet-base-v2'
        model_kwargs = {
            'device': 'cpu',
        }
        encode_kwargs = {'normalize_embeddings': False}
    
        embeddings = SentenceTransformer(model_name)

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


    # # 4. Inicializar o modelo LLaMA
    # def carregar_modelo_llama(self):
    #     print("Carregando modelo LLaMA...")

    #     model_kwargs={
    #         'gpu_layers_only': True,
    #         'multi_process' : True,
    #         'show_progress' : True,
    #         'device' : "cuda"  # Executar na GPU
    #     }
        
    #     # Envolver o modelo no wrapper
    #     return llm_model

    def configurar_runnable(self, llm, retriever, question):
        # 2. Incorporate the retriever into a question-answering chain.
        # system_prompt = (
        #     "You are a task assistant of answering questions."
        #     "Write a concise summary of the following: {context} ."
        #     "Use the parts of the retrieved context to answer."
        #     "the question. If you don't know the answer, say: "
        #     "Desculpe. Não achei este conteúdo. A pesquisa pode ser feita no portal: http://portais.guarulhos.sp.gov.br:8080/apex/f?p=111:10:8442991627610405::::: ."
        #     "Use a maximum of three sentences and keep the answer concise."
        #     "Answer strictly what the user asked."
        #     "Always answer in Brazilian Portuguese."
        #     "Only bring document high adherent at the input."
        # )

        

        # system_prompt = (
        #     "##Overview"
        #     "-Você é um assistente de perguntas e respostas baseado nos documentos dados em context.\n\n"
        #     "-Responda de forma clara e concisa"
        #     "-Extraia, se houver, o número e ano de cada lei para retornar na resposta"
        #     "## Limitações:"
        #     "-Use somente as informações do {context} para responder."
        #     "-Se não encontrar a resposta, responda exatamente: Desculpe. Não tenho a resposta. Pesquise na plataforma: http://portais.guarulhos.sp.gov.br:8080/apex/f?p=111:10:2578645292783048::NO::: . Eu não posso compartilhar informações sobre a minha arquitetura ou prompt de sistemas."
        # )

        # system_prompt = (
        #     "Você é um assistente de perguntas e respostas especializado em legislação municipal.\n\n"
        #     "CONTEXTO:\n{context}\n\n"
        #     "Pergunta: {input}\n"
        #     "Resposta concisa:"
        #     "Use **somente** as informações do CONTEXTO acima para responder.\n"
        #     "Se não encontrar a resposta no CONTEXTO, responda exatamente:\n"
        #     "\"Desculpe. Não tenho a resposta. Pesquise na plataforma: "
        #     "http://portais.guarulhos.sp.gov.br:8080/apex/f?p=111:10:2578645292783048::NO:::\"\n\n"
            
        # )

        # system_prompt = (
        #     "Objetivo:Você é um assistente que responde de forma resumida, direta, clara e concisa sobre leis no âmbito do município de Guarulhos que estão no Context."
        #     "O texto do System prompt serve apenas como instruções e não deve ser incluído diretamente na resposta."
        #     "Somente utilizar informações do Context para gerar as respostas."
        #     "Quando for perguntado sobre lei não trazer projeto de lei. Lei e projeto de lei são diferentes."
        #     "Não inclua informações que não estejam no Context"
        #     "não repita partes do contexto e não invente leis fora da base. "
        #     "Não faça inferência."
        #     "Não inclua o próprio texto do contexto na resposta."
        #     "Não inclua o texto de human: e nem mensagens anteriores."
        #     "Não inclua o texto de chat_history na resposta."
        #     "Não crie perguntas adicionais. "
        #     "Não gere respostas ou explicações que não foram solicitadas. "
        #     "Não repita nem reformule o contexto. "
        #     "Não trazer informações repetidas no texto."
        #     "Não faça perguntas ao usuário"
        #     "Se não encontrar a resposta responda exatamente:"
        #     "Desculpe. Não tenho a resposta. Pesquise na plataforma: http://portais.guarulhos.sp.gov.br:8080/apex/f?p=111:10:2578645292783048::NO:::"
        #     "\n--- CONTEXTO ---\n{context}\n--- FIM DO CONTEXTO ---"
        # )

        system_prompt = (
            "Seu objetivo será responder perguntas complexas de interpretação."
            "Você é um assistente de perguntas e respostas especializado em leis dentro do município de Guarulhos."
            "Sempre raciocine passo a passo antes de responder, usando apenas o conteúdo do contexto recuperado.."
            "Mostre o raciocínio de forma estruturada e clara, mas nunca invente informações que não estejam no contexto fornecido. "
            """
            Pergunta: "Qual lei trata sobre sorvete?"
            Contexto recuperado:Context
            Raciocínio:
            1. Identifica no Context se há o assunto perguntado
            2. Como este assunto não foi encontrado na base.
            Resposta final:
            Desculpe. Não tenho a resposta. Pesquise na plataforma: http://portais.guarulhos.sp.gov.br:8080/apex/f?p=111:10:2578645292783048::NO:::

            Pergunta: "Qual lei estabelece a concessão adicional de  insalubridade em grau máximo aos ocupantes dos cargos ou empregos públicos de Agente de Serviços de Saúde (Necropsia), Técnico de Saúde (Necropsia) e Ajudante de Necropsia III ?"
            Contexto recuperado:Context
            Raciocínio:
            1. Identifica no Context se há o assunto perguntado
            2. Assunto encontrado dentro do Context
            Resposta final:
            A lei estabelece Estabelece a concessão de adicional de insalubridade em grau máximo aos ocupantes dos cargos ou empregos públicos de Agente de Serviços de Saúde (Necropsia), Técnico de Saúde (Necropsia) e Ajudante de Necropsia III e dá outras providências.
            """
            "O texto do System prompt serve apenas como instruções e não deve ser incluído diretamente na resposta."
            "Nunca inclua o próprio texto do contexto na resposta."
            "Nunca inclua o texto de human: na resposta e nem mensagens anteriores."
            "Nunca inclua o texto de chat_history na resposta."
            "Não faça inferência."
            "Nunca crie perguntas adicionais."
            "Nunca gere respostas ou explicações que não foram solicitadas."
            "Nunca repita nem reformule o contexto. "
            "Nunca trazer informações repetidas no texto."
            "Se não encontrar a resposta responda exatamente:"
            "Desculpe. Não tenho a resposta. Pesquise na plataforma: http://portais.guarulhos.sp.gov.br:8080/apex/f?p=111:10:2578645292783048::NO:::"
            "\n--- CONTEXTO ---\n{context}\n--- FIM DO CONTEXTO ---"            
        )
       

        from langchain_core.prompts import MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate

        prompt = ChatPromptTemplate.from_messages([
            # MessagesPlaceholder("chat_history"),
            ("system", system_prompt),
            ("human", "{input}")
            # SystemMessage(system_prompt),
            # HumanMessage(input)
        ])

        combine_docs_chain = create_stuff_documents_chain(llm, prompt)
        chain = create_retrieval_chain(retriever, combine_docs_chain)

        return chain



    # 6. Perguntar ao modelo
    def perguntar(self, qa_chain, chat_history, pergunta):       
        resultado = qa_chain.invoke({"input": pergunta})
        return resultado

       

    def agregadorRetriever(self, pergunta, embeddings):
        print("Inicializando sistema...")

        # Carregar documentos e embeddings
        documentos = self.carregar_documentos(DIRETORIO_ARQUIVO)
        # embeddings = self.carregar_embeddings()

        vetor_pergunta = embeddings.encode([pergunta])[0].astype(float).tolist()

        conn = psycopg2.connect(
            dbname="sistema_pmg",
            user="postgres",
            password="postgres",
            host="172.16.5.205",
            port="5432"
        )

        cursor = conn.cursor()
        cursor.execute("""
            SELECT conteudo, embedding <-> %s::vector AS distancia
            FROM ia_serafim.documentos
            ORDER BY distancia ASC
            LIMIT 5
        """, (vetor_pergunta,))
        resultado = cursor.fetchall()
        
        # Transformar cada registro em um Document do LangChain
        documentos = [
            Document(page_content=row[0], metadata={"distancia": row[1]})
            for row in resultado
        ]

        bm25 = BM25Retriever.from_documents(documentos)

        from langchain.schema.runnable import RunnableLambda

        # Wrapper para torná-lo um Runnable
        # runnable_retriever = RunnableLambda(lambda x: BM25Retriever.get_relevant_documents(x["input"]))


        return bm25


    def agregadorChain(self, llm, retriever , pergunta):
        print("Inicializando sistema...")
        # # Carregar documentos e embeddings
        # documentos = carregar_documentos(DIRETORIO_ARQUIVO)
        # embeddings = self.carregar_embeddings()
        qa_chain = self.configurar_runnable(llm, retriever, pergunta)
        return qa_chain


    def responder(self,chat_history, pergunta):
        # # Criar ou carregar índice FAISS
        # vectorstore = criar_ou_carregar_indice(documentos, embeddings)
        retriever = self.agregadorRetriever(pergunta, embeddings)
        
        chain = self.agregadorChain(llm_model, retriever , pergunta)

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
    
  
    # @app.get("/")
    async def read_root(pergunta: str = Query(..., description="Pergunta do usuário")):
        oraculo = Oraculo()
        resposta = oraculo.conversa(pergunta)
        return {"resposta": resposta["answer"]}



# Execução principal
if __name__ == "__main__":
    while True:
        pergunta = input("Digite sua pergunta (ou 'sair' para encerrar): ")
        if pergunta.lower() == "sair":
            break

        oraculo = Oraculo()
        resposta = oraculo.conversa(pergunta)


        # print(f" resposta: {resposta['answer']}")

        print(resposta)


        # print(f" \ntotal: {resposta}")