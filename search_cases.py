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
    ServiceContext,
    VectorStoreIndex
)
from llama_index.retrievers import VectorIndexRetriever
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores import PineconeVectorStore
from langchain.chat_models import ChatOpenAI
from llama_index.vector_stores.types import ExactMatchFilter, MetadataFilters
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
        doc.metadata['content_type'] = "case_summary"
    return docs


def test_index(docs):
    
    index = VectorStoreIndex.from_documents(docs)
    filters = MetadataFilters(filters=[
        ExactMatchFilter(
            key="case_number",
            value="DCPI003618_2019"
        )]
    )
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=3,
        vector_store_query_mode="default"
    )

    filtered_nodes = []
    
    nodes = retriever.retrieve("my client slips and falls. Find me cases.")

    for node in nodes:
        matches = True
        for f in filters.filters:
            if f.key not in node.metadata:
                matches = False
                continue
            if f.value != node.metadata[f.key]:
                matches = False
                continue
        if matches:
            filtered_nodes.append(node)

    # print("\n\n")
    # print("nodes: ")
    # print(nodes)
    # print("\n\n")
    # print("filtered_nodes: ")
    # print(filtered_nodes)



def build_context(model_name):
    llm_predictor = LLMPredictor(
        llm = ChatOpenAI(temperature=0, model_name=model_name)
    )
    return ServiceContext.from_defaults(llm_predictor=llm_predictor)


def build_summaries_index(docs):
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
        print("Pinecone index not exist. Need to create one.")
    pinecone_index = pinecone.Index(index_name)
    print("Pinecone index has already been created and now connected.")

    service_context = build_context("gpt-3.5-turbo")

    #for all summaries, create one instance of PineconeVectorStore and then GPTVectorStoreIndex
    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index
    )
    
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    final_summary_index = GPTVectorStoreIndex.from_documents(docs, storage_context=storage_context, service_context=service_context)
    
    print("vector store built and stored in Pinecone.")
    return final_summary_index


def query_search_engine(query):

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
        print("Pinecone index not exist. Need to create one.")
    pinecone_index = pinecone.Index(index_name)
    print("Pinecone index has already been created and now connected.")

    #construct a vector store from Pinecone
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index,
        metadata_filters={"content_type": "case_summary"})
    
    #create a VectorStoreIndex from the existing vector store in Pinecone and then query it
    vector_index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        # metadata_filters={"content_type": "case_summary"}
        )

    # #create a retriever from the vector_index
    # retriever = vector_index.as_retriever(
    #     similarity_top_k=1000
    # )

    #define metadata filters and filter the nodes
    filters = MetadataFilters(filters=[
        ExactMatchFilter(
            key="content_type",
            value="case_summary"
        )]
    )

    retriever = VectorIndexRetriever(
        index = vector_index,
        similarity_top_k=10,
        vector_store_query_mode="default",
        filters=filters
    )

    # #retrieve nodes
    nodes = retriever.retrieve(query)
    print("1. Number of nodes retrieved: ", len(nodes))
    
    #check how many nodes are unique
    all_node_text = []
    for node in nodes:
        node_text = node.text.replace("\n", "")
        if node_text not in all_node_text:
            all_node_text.append(node_text)
    
    print("2. number of nodes: ",len(nodes))
    print("3. number of unique nodes: ",len(all_node_text))

    #######HACKY WAY#######
    # #define metadata filters and filter the nodes
    # filters = MetadataFilters(filters=[
    #     ExactMatchFilter(
    #         key="content_type",
    #         value="case_summary"
    #     )]
    # )

    # filtered_nodes = []

    # for node in nodes:
    #     matches = True
    #     for f in filters.filters:
    #         if f.key not in node.metadata:
    #             matches = False
    #             continue
    #         if f.value != node.metadata[f.key]:
    #             matches = False
    #             continue
    #     if matches:
    #         filtered_nodes.append(node)
    # for filtered_node in filtered_nodes:
    #     print(filtered_node.metadata)
    # print("number of nodes: ",len(filtered_nodes))
    # print("\n\n")
    #######HACKY WAY#######
    
    #extract the case number from the filtered nodes and then de-duplicate them
    raw_search_results = []
    dedup_search_results = []

    for node in nodes:
        raw_search_results.append(node.metadata['case_number'])
    print("4. RSR: ",raw_search_results)
    [dedup_search_results.append(raw_search_result) for raw_search_result in raw_search_results if raw_search_result not in dedup_search_results]
    print("5. DRSR: ",dedup_search_results)
    print("\n")
    print(f"hahaah! There are {len(dedup_search_results)} unique search results!")
    return dedup_search_results
    
# query = "my client slips and falls in a shopping mall."
# dedup_search_results = query_search_engine(query)
# final_answers = query_case(dedup_search_results, query)
# print(f"7. FINAL: {final_answers}")


if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            dedup_search_results = query_search_engine(prompt)
            final_answers = query_case(dedup_search_results, prompt)
            st.write(final_answers)
            message = {"role": "assistant", "content": final_answers}
            st.session_state.messages.append(message)