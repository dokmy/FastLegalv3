import dotenv
import os
import streamlit as st
import logging
import sys
from dotenv import load_dotenv
import openai
import pinecone
from llama_index import (
    SimpleDirectoryReader,
    GPTVectorStoreIndex,
    LLMPredictor,
    ServiceContext
)
from llama_index.retrievers import VectorIndexRetriever
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores import PineconeVectorStore
from langchain.chat_models import ChatOpenAI
from case_query import query_case

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

st.header("Smart Search Engine for Legal Cases.")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant", 
         "content": "Describe your case and we will find you relevant cases."
         }
    ]

@st.cache_resource(show_spinner=False)

def create_list_of_case_numbers():
    list_of_case_numbers = []
    
    folder_path = './summaries'
    all_summaries = os.listdir(folder_path)

    for summary in all_summaries:
        case_number = summary.replace(".txt","")
        list_of_case_numbers.append(case_number)
    return list_of_case_numbers


def build_docs(list_of_case_numbers):
    docs = SimpleDirectoryReader(input_dir='./summaries', filename_as_id=True).load_data()
    # print(f"There are {len(docs)} Document Objects.\n")
    # print("\nHere are the doc 1")
    # print(docs[0])
    # print("\nHere are the doc 2")
    print("Here are the number of docs:", len(docs))
    for doc in docs:
        case_number_with_ext = doc.doc_id.replace('summaries/',"")
        case_number = case_number_with_ext.replace(".txt","")
        doc.metadata['case_number'] = case_number
        # print(doc.metadata)
    return docs


def build_context(model_name):
    llm_predictor = LLMPredictor(
        llm = ChatOpenAI(temperature=0, model_name=model_name)
    )
    return ServiceContext.from_defaults(llm_predictor=llm_predictor)

def build_summaries_index(list_of_case_numbers,docs):
    #init Pinecone and connect with Pinecone Index, then create service context
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENVIRONMENT")
    )

    index_name = "cases-index"
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            index_name,
            dimension=1536,
            metric="cosine"
        )
    pinecone_index = pinecone.Index(index_name)
    print("Canvas is connected.")

    service_context = build_context("gpt-3.5-turbo")

    #for all summaries, create one instance of PineconeVectorStore and then GPTVectorStoreIndex
    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index
    )
    
    index = GPTVectorStoreIndex.from_documents(docs)

    return index

def query_search_engine(query):

    list_of_case_numbers = create_list_of_case_numbers()
    docs = build_docs(list_of_case_numbers=list_of_case_numbers)
    index = build_summaries_index(list_of_case_numbers, docs)
    retiever = VectorIndexRetriever(
        index=index,
        similarity_top_k=5
    )
    nodes = retiever.retrieve(query)

    search_results = []
    retrieved_cases = []
    for node in nodes:
        retrieved_cases.append(node.metadata['case_number'])

    # print("Here are the retrieved nodes: ", retrieved_cases)
    [search_results.append(case_num) for case_num in retrieved_cases if case_num not in search_results]

    # print("Here are the final results: ",results)
    return search_results
    
if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            search_results = query_search_engine(prompt)

            responses = [query_case(case_number) for case_number in search_results]

            for response in responses:
                st.write(response)

            message = {"role": "assistant", "content": responses}
            st.session_state.messages.append(message)
            # for search_result in search_results:
            #     final_answer = query_case(search_result, prompt)
            #     # st.markdown("- " + search_result)
            #     st.write(final_answer)
            # message = {"role": "assistant", "content": search_results}
            # st.session_state.messages.append(message)