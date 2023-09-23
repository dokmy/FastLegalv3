import logging
import sys
import os
import openai
from dotenv import load_dotenv, dotenv_values
import streamlit as st
from llama_index import SimpleDirectoryReader, LLMPredictor, get_response_synthesizer, SummaryIndex, VectorStoreIndex, ServiceContext, Document
from llama_index.indices.document_summary import DocumentSummaryIndex
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.schema import IndexNode
from llama_index.llms import OpenAI
from llama_index.agent import OpenAIAgent
from llama_index.retrievers import RecursiveRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.response_synthesizers import get_response_synthesizer

load_dotenv()
#openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = st.secrets["OPENAI_API_KEY"]

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

st.header("Ask me about legal stuff!!!")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant", 
         "content": "Ask me a question about the laws of HK!"
         }
    ]


@st.cache_resource(show_spinner=False)

#1. create List of Index Node Objects
def create_list_of_Index_Nodes():
    
    files = os.listdir('./data')
    list_of_index_nodes = []

    for file in files:
        case_number = os.path.splitext(file)[0]
        file_path = os.path.join('./data', file)
        
        #make summary
        docs = SimpleDirectoryReader(input_files=[file_path]).load_data()
        for doc in docs:
            doc.text = doc.text.replace("\n \n  A \n \n \n \nB \n \n \n \nC \n \n \n \nD \n \n \n \nE \n \n \n \nF \n \n \n \nG \n \n \n \nH \n \n \n \nI \n \n \n \nJ \n \n \n \nK \n \n \n \nL \n \n \n \nM \n \n \n \nN \n \n \n \nO \n \n \n \nP \n \n \n \nQ \n \n \n \nR \n \n \n \nS \n \n \n \nT \n \n \n \nU \n \n \n \nV A \n \n \n \nB \n \n \n \nC \n \n \n \nD \n \n \n \nE \n \n \n \nF \n \n \n \nG \n \n \n \nH \n \n \n \nI \n \n \n \nJ \n \n \n \nK \n \n \n \nL \n \n \n \nM \n \n \n \nN \n \n \n \nO \n \n \n \nP \n \n \n \nQ \n \n \n \nR \n \n \n \nS \n \n \n \nT \n \n \n \nU \n \n \n \nV", "")
        index = VectorStoreIndex.from_documents(docs)
        query_engine = index.as_query_engine()
        summary = query_engine.query("Please summarise this case for me.")
        index_node = IndexNode(text=summary.response, index_id=case_number)
        list_of_index_nodes.append(index_node)
    
    return list_of_index_nodes

#2. Create Dictionary of Agents
def create_dict_of_agents():

    files = os.listdir('./data')
    
    agents = {}

    llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
    service_context = ServiceContext.from_defaults(llm=llm)

    for file in files:
        file_path = os.path.join('./data', file)
        case_number = os.path.splitext(file)[0]
        docs = SimpleDirectoryReader(input_files=[file_path]).load_data()
        for doc in docs:
            doc.text = doc.text.replace("\n \n  A \n \n \n \nB \n \n \n \nC \n \n \n \nD \n \n \n \nE \n \n \n \nF \n \n \n \nG \n \n \n \nH \n \n \n \nI \n \n \n \nJ \n \n \n \nK \n \n \n \nL \n \n \n \nM \n \n \n \nN \n \n \n \nO \n \n \n \nP \n \n \n \nQ \n \n \n \nR \n \n \n \nS \n \n \n \nT \n \n \n \nU \n \n \n \nV A \n \n \n \nB \n \n \n \nC \n \n \n \nD \n \n \n \nE \n \n \n \nF \n \n \n \nG \n \n \n \nH \n \n \n \nI \n \n \n \nJ \n \n \n \nK \n \n \n \nL \n \n \n \nM \n \n \n \nN \n \n \n \nO \n \n \n \nP \n \n \n \nQ \n \n \n \nR \n \n \n \nS \n \n \n \nT \n \n \n \nU \n \n \n \nV", "")

        #create vector index and query engine
        vector_index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        vector_query_engine = vector_index.as_query_engine()


        #create summary index and query engine
        summary_index = SummaryIndex.from_documents(docs, service_context=service_context)
        summary_index_engine = summary_index.as_query_engine()

        #create Query Engine Tool Objects from the two indexes and group them to a list called query_engine_tools
        query_engine_tools = [
            QueryEngineTool(
                query_engine=vector_query_engine,
                metadata=ToolMetadata(
                    name="vector_tool",
                    description=f"Useful for retrieving specific context from this legal case with case number: {case_number}"
                )
            ),
            QueryEngineTool(
                query_engine=summary_index_engine,
                metadata=ToolMetadata(
                    name='summary_tool',
                    description=f"Useful for summarization questions related to this legal case with case number: {case_number}"
                )
            )
        ]

        #make an agent from the list called query_engine_tools
        function_llm = OpenAI(model="gpt-3.5-turbo-0613")
        agent = OpenAIAgent.from_tools(
            query_engine_tools,
            llm=function_llm,
            verbose=True
        )

        #put the agent into the dict of agents
        agents[case_number] = agent
    
    return agents


#3. Create the final query engine
def create_final_query_engine():
    
    #define top-level retriever
    list_of_index_nodes = create_list_of_Index_Nodes()
    top_level_vector_store = VectorStoreIndex(list_of_index_nodes)
    top_level_vector_store_retriever = top_level_vector_store.as_retriever(similarity_top_k=1)

    #define agnets
    agents = create_dict_of_agents()

    #define recursive_retriever for the final_query_engine
    recursive_retriever = RecursiveRetriever(
        "vector",
        retriever_dict = {"vector": top_level_vector_store_retriever},
        query_engine_dict=agents,
        verbose=True
    )

    #define final_query_engine
    response_synthesizer = get_response_synthesizer(
        response_mode="compact"
    )
    llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
    service_context = ServiceContext.from_defaults(llm=llm)
    final_query_engine = RetrieverQueryEngine.from_args(
        recursive_retriever,
        response_synthesizer=response_synthesizer,
        service_context=service_context
    )

    return final_query_engine

final_query_engine = create_final_query_engine()
# response = final_query_engine.query("Summarise this legal case with case number: DCPI002019_2013")
# print(response)


if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = final_query_engine.query(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)