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
import sys

load_dotenv()
openai.api_key = st.secrets["OPENAI_API_KEY"]

# if len(sys.argv) < 2:
#     print("No argument")
#     sys.exit(1)

# url_segment = sys.argv[1]
# print(f"Recieved Url sement: {url_segment}")

def create_list_of_case_numbers(cases_folder_path):
    list_of_case_numbers = []

    for sub_folder in os.listdir(cases_folder_path):
        sub_folder_path = os.path.join(cases_folder_path, sub_folder)
        cases = os.listdir(sub_folder_path)
        for case in cases:
            case_number = case.replace(".docx","")
            list_of_case_numbers.append(case_number)
    return list_of_case_numbers


def build_case_query_engine(case_num):

    pinecone.init(
        api_key=st.secrets["PINECONE_API_KEY"],
        environment=st.secrets["PINECONE_ENVIRONMENT"]
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
        )
    ])

    # PROMPT_TEMPLATE = (
    #     "Here are the context information:"
    #     "\n------------------------------\n"
    #     "{context_str}"
    #     "\n------------------------------\n"
    #     "You are a AI legal assistant for lawyers in Hong Kong. Answer the follwing question in two parts. Break down these two parts with sub-headings. First, explained what happened in the case for reference in the context. Second, explain how this case is relevant to the following siutation or question: {query_str}. \n"
    #     )

    PROMPT_TEMPLATE = (
        "Here are the context information:"
        "\n---------------------------------\n"
        "{context_str}"
        "\n---------------------------------\n"
        "You are a AI legal assistant for lawyers in Hong Kong. Answer your question based on the context given and you must mention the exact sentences or paragraphs you used to return the answer of this question: {query_str}"
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


def build_context(model_name):
    llm_predictor = LLMPredictor(
        llm=ChatOpenAI(temperature=0, model_name=model_name)
        )
    return ServiceContext.from_defaults(llm_predictor=llm_predictor)


def query_case(case_num, query):
    query_engine = build_case_query_engine(case_num)
    response = query_engine.query(query)
    # print("HAHAH")
    # print(response.get_formatted_sources().strip())

    return response.response_gen


def main():
    params = st.experimental_get_query_params()
    
    if params:
        incoming = params['case_num'][0]
    else:
        incoming = 0

    st.set_page_config(
        page_title=":robot_face:Chat with Any Case",
        page_icon=":robot_face:",
        layout="centered",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://www.extremelycoolapp.com/help',
            'Report a bug': "https://www.extremelycoolapp.com/bug",
            'About': "# Find me at adrien@stepone.agency"
            }
    )
    
    st.title(":robot_face: Chat with Any Case")

    cases_folder_path = "./data/DCPI"
    list_of_case_numbers = create_list_of_case_numbers(cases_folder_path)
    
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": f"Ask me a question about this case!"}
        ]

    if "streaming" not in st.session_state:
        st.session_state.streaming = False


    if "selected_case" not in st.session_state:
        st.session_state.selected_case = None
    

    if prompt := st.chat_input("Your question"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.streaming = True

    if st.session_state.streaming:
        st.warning("Please wait for the current answer to complete.")
        st.selectbox(
        'Which case would you like to chat with?',
        ("Please wait...",), disabled=True)
        case_num = st.session_state.selected_case
    else:
        if st.session_state.selected_case != None:
            index = list_of_case_numbers.index(st.session_state.selected_case)
        else:
            if incoming != 0:
                index = list_of_case_numbers.index(incoming)
            else:
                index = 0

        case_num = st.selectbox(
        'Which case would you like to chat with?',
        (list_of_case_numbers), index = index)
        
        if st.session_state.selected_case != case_num:
            st.session_state.selected_case = case_num
            st.session_state.messages = [
                {"role": "assistant", "content": f"Ask me a question about case {case_num}."}
            ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Generating answer..."):
                response = query_case(case_num, prompt)
                ans_box = st.empty()
                stream = []
                for res in response:
                    stream.append(res)
                    answer = "".join(stream).strip()
                    ans_box.markdown(answer)
                message = {"role": "assistant", "content": answer}
                st.session_state.messages.append(message)
                st.session_state.streaming = False
                st.experimental_rerun()
    

if __name__ == "__main__":
    main()