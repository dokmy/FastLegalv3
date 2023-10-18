import os
import dotenv
from dotenv import load_dotenv
import openai
from llama_index import(
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    ServiceContext,
    LLMPredictor,
    GPTVectorStoreIndex,
    QuestionAnswerPrompt
)
import pinecone
from llama_index.vector_stores import PineconeVectorStore
from llama_index.retrievers import VectorIndexRetriever
from langchain.chat_models import ChatOpenAI
from llama_index.vector_stores.types import ExactMatchFilter, MetadataFilters
import streamlit as st

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def create_list_of_case_numbers(cases_folder_path):
    list_of_case_numbers = []
    cases = os.listdir(cases_folder_path)
    for case in cases:
        case_number = case.replace(".docx","")
        list_of_case_numbers.append(case_number)
    print(list_of_case_numbers)
    return list_of_case_numbers


def build_docs(cases_folder_path, content_type):
    docs = []
    docs = SimpleDirectoryReader(input_dir=cases_folder_path).load_data()
    for doc in docs:
        # print(doc.metadata)
        fn = doc.metadata["file_name"]
        case_num = fn.replace(".docx","")
        doc.metadata = {
            "content_type": content_type,
            "fn": fn,
            "case_num": case_num
            }
    print(f"Docs created. Number of docs: {len(docs)}")
    return docs


def build_context(model_name):
    llm_predictor = LLMPredictor(
        llm=ChatOpenAI(temperature=0, model_name=model_name)
        )
    return ServiceContext.from_defaults(llm_predictor=llm_predictor)


def upsert_docs(docs):
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENVIRONMENT")
    )

    index_name = "cases-index"
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            index_name,
            dimension=1536,
            metric='cosine'
        )
        print("Pinecone canvas does not exist. Just created and connected.")
    pinecone_index = pinecone.Index(index_name)
    print("Pinecone canvas already exists. Now we're connected.")

    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    service_context = build_context("gpt-3.5-turbo")

    GPTVectorStoreIndex.from_documents(
            docs,
            storage_context=storage_context,
            service_context=service_context
    )

    print("Upsert to Pinecone done.")


def delete_vectors():
    
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENVIRONMENT")
    )

    index_name = "cases-index"
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            index_name,
            dimension=1536,
            metric='cosine'
        )
        print("Pinecone canvas does not exist. Just created and connected.")
    pinecone_index = pinecone.Index(index_name)
    print("Pinecone canvas already exists. Now we're connected.")

    #delete by metadata
    # pinecone_index.delete(
    #     filter={
    #         "content_type": "case_itself"
    #     }
    # )

    #delete everything
    # delete_response = pinecone_index.delete(delete_all=True)
    

def count_by_metadata(content_type):
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENVIRONMENT")
    )

    index_name = "cases-index"
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            index_name,
            dimension=1536,
            metric='cosine'
        )
        print("Pinecone canvas does not exist. Just created and connected.")
    pinecone_index = pinecone.Index(index_name)
    print("Pinecone canvas already exists. Now we're connected.")

    stats = pinecone_index.describe_index_stats(filter={"content_type": content_type})


    print(stats)


print("Before upseting...")
content_type = "RUL"
count_by_metadata(content_type)

cases_folder_path = "/Users/adrienkwong/Downloads/FastLegal files/FastLegal - LlamaIndex + Streamlit/data/DCPI/ruling"
docs = build_docs(cases_folder_path, content_type)
# upsert_docs(docs)

print("After upseting...")
count_by_metadata(content_type)
