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

# load_dotenv()
openai.api_key = st.secrets("OPENAI_API_KEY")

def create_list_of_case_numbers(cases_folder_path):
    list_of_case_numbers = []
    cases = os.listdir(cases_folder_path)
    for case in cases:
        case_number = case.replace(".docx","")
        list_of_case_numbers.append(case_number)
    print(list_of_case_numbers)
    return list_of_case_numbers


def build_docs(cases_folder_path):
    docs = []
    docs = SimpleDirectoryReader(input_dir=cases_folder_path).load_data()
    for doc in docs:
        # print(doc.metadata)
        fn = doc.metadata["file_name"]
        case_num = fn.replace(".docx","")
        doc.metadata = {
            "content_type":"case_itself",
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
        api_key=st.secrets("PINECONE_API_KEY"),
        environment=st.secrets("PINECONE_ENVIRONMENT")
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


def build_search_engine():

    pinecone.init(
        api_key=st.secrets("PINECONE_API_KEY"),
        environment=st.secrets("PINECONE_ENVIRONMENT")
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

    vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    filters = MetadataFilters(filters=[
        ExactMatchFilter(
            key = "content_type",
            value = "case_itself"
        )
    ])

    retriever = VectorIndexRetriever(
        index=vector_index,
        similarity_top_k=3,
        vector_store_query_mode="default",
        filters=filters
    )
    print("Top Level Search Engine Retriever created.")
    return retriever


def query_search_engine(retriever, query):
    
    nodes = retriever.retrieve(query)
    print(f"Nodes retrieved. Number of nodes: {len(nodes)}")

    case_num_in_nodes = []
    for node in nodes:
        try:
            case_num = node.metadata["case_num"]
            if case_num not in case_num_in_nodes:
                case_num_in_nodes.append(case_num)
        except:
            print("This node does not have case_num.")
    print(f"Here are the unique case numbers: {case_num_in_nodes}")
    print(f"Unique case numbers retrieved: {len(case_num_in_nodes)}")
    return case_num_in_nodes


def build_case_query_engine(case_num):
    pinecone.init(
        api_key=st.secrets("PINECONE_API_KEY"),
        environment=st.secrets("PINECONE_ENVIRONMENT")
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

    vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    filters = MetadataFilters(filters=[
        ExactMatchFilter(
            key = "case_num",
            value = case_num
        ),
        ExactMatchFilter(
            key = "content_type",
            value = "case_itself"
        )
    ])

    PROMPT_TEMPLATE = (
        "Here are the context information:"
        "\n------------------------------\n"
        "{context_str}"
        "\n------------------------------\n"
        "You are a AI legal assistant for lawyers in Hong Kong. Answer the follwing question in two parts. Break down these two parts with sub-headings. First, explained what happened in the case for reference in the context. Second, explain how this case is relevant to the following siutation or question: {query_str}. \n"
        )
    
    QA_PROMPT = QuestionAnswerPrompt(PROMPT_TEMPLATE)

    query_engine = vector_index.as_query_engine(
        similarity_top_k=3,
        vector_store_query_mode="default",
        filters=filters,
        text_qa_template=QA_PROMPT,
        streaming = True,
        service_context=build_context("gpt-3.5-turbo")
    )
    print("Query engine created.")
    return query_engine


def query_case(case_num, query):
    query_engine = build_case_query_engine(case_num)
    response = query_engine.query(query)
    # response.print_response_stream()
    # for res in response.response_gen:
    #     print(res)
    return response.response_gen


# query = "My client was speeding and hit a jaywalker."
# retriever = build_search_engine()
# list_of_case_num = query_search_engine(retriever, query)
# for case_num in list_of_case_num:
#     query_case(case_num, query)


st.sidebar.title("Search Legal Cases")

user_input = st.sidebar.text_area("Describe your client's siutation:")
submit_button = st.sidebar.button("Search")


if submit_button:
    with st.spinner('Generating answers...'):
        query = user_input
        retriever = build_search_engine()
        list_of_case_num = query_search_engine(retriever, query)
        st.markdown(f"**Found {len(list_of_case_num)} cases. Here are the search results:**")
        for case_num in list_of_case_num:
            # st.markdown(f"## {case_num}")
            ans_box = st.empty()
            box_id = "custom_ans_box"
            st.markdown(
                f"""
                 <style>
                    #{box_id} {{
                        background-color: rgba(255, 165, 0, 0.5); 
                        padding: 15px;
                        border-radius: 10px;
                    }}
                </style>
                """,
                unsafe_allow_html=True,
            )

            stream = []
            for res in query_case(case_num, query):
                stream.append(res)
                answer = "".join(stream).strip()
                ans_box.markdown(
                    f'<div id="{box_id}"><h2>{case_num}</h2><br>{answer}</div>', 
                    unsafe_allow_html=True
                                )
                # ans_box.markdown(f"{answer}")

            






#ONLY RUN WHEN THERE'RE NEW DOCS TO UPSERT
# cases_folder_path = "./data/judgements_docx"
# docs = build_docs(cases_folder_path) 
# upsert_docs(docs)

