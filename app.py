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
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.header("Ask me about legal stuff!!!")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant", 
         "content": "Ask me a question about the laws of HK!"
         }
    ]

#folder_path = "/Users/adrienkwong/Downloads/FastLegal files/FastLegal - LlamaIndex + Streamlit/data"

folder_path = './data'

@st.cache_resource(show_spinner=False)

#create IndexNode for each case
def create_summaries():
    cases_summaries = {}
    nodes = []
    
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        case_number = os.path.splitext(os.path.basename(file_path))[0]
        try:
            docs = SimpleDirectoryReader(input_files=[file_path]).load_data()
            index = VectorStoreIndex.from_documents(docs)
            query_engine = index.as_query_engine()
            summary = query_engine.query("The docuemtns are all about a legal case in Hong Kong. Please summarise the documents. Start by identifying if this case is about work injury, traffic accident or other injuries. Then create at least 10 learnings from this case that can be applied to other legal cases.")
            cases_summaries[case_number] = summary
            node = IndexNode(text=str(summary), index_id=case_number)
            nodes.append(node)
        except Exception as e:
             print(f"Error reading {file_path}. Error: {e}")

    return nodes


#Create document agent over each case
def create_agents():
    agents = {}

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        case_number = os.path.splitext(os.path.basename(file_path))[0]
        #load each pdf into multiple document objects
        docs = SimpleDirectoryReader(input_files=[file_path]).load_data()
        #build vector index
        vector_index = VectorStoreIndex.from_documents(docs)
        #build summary index
        summary_index = SummaryIndex.from_documents(docs)
        #define query engines
        vector_query_engine = vector_index.as_query_engine()
        list_query_engine = summary_index.as_query_engine()

        #define tools
        query_engine_tools = [
            QueryEngineTool(
                query_engine=vector_query_engine,
                metadata=ToolMetadata(
                    name='vector_tool',
                    description=f"Userful for summarization questions related to {case_number}"
                )
            ),
            QueryEngineTool(
                query_engine=list_query_engine,
                metadata=ToolMetadata(
                    name='summary_tool',
                    description=f"Userful for retrieving specific context from {case_number}"
                )
            )
        ]

        #build agent
        function_llm = OpenAI(model='gpt-3.5-turbo-0613')
        agent = OpenAIAgent.from_tools(
            query_engine_tools,
            llm=function_llm,
            verbose=True
        )

        agents[case_number] = agent
        return agents
    

def create_final_query_engine():

    nodes = create_summaries()
    agents = create_agents()

    #define top-level retriever
    vector_index = VectorStoreIndex(nodes)
    vector_retriever = vector_index.as_retriever(similarity_top_k=3)

    recursive_retriever = RecursiveRetriever(
    "vector",
    retriever_dict={"vector": vector_retriever},
    query_engine_dict = agents,
    verbose=True
    )

    llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
    service_context = ServiceContext.from_defaults(llm=llm)

    response_synthesizer = get_response_synthesizer(
        # service_context = service_context,
        response_mode = "tree_summarize"
    )

    query_engine = RetrieverQueryEngine.from_args(
        recursive_retriever,
        response_synthesizer=response_synthesizer,
        service_context=service_context
    )

    return query_engine


chat_engine = create_final_query_engine()

if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.query(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)