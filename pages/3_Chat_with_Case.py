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
import json
from datetime import datetime
import time


load_dotenv("./.env")

try:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")

except:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    pinecone_api_key = st.secrets["PINECONE_API_KEY"]
    pinecone_environment = st.secrets["PINECONE_ENVIRONMENT"]



        

@st.cache_data
def get_all_case_nos():
    all_action_nos = []

    all_jsonl_files = os.listdir("./all_jsonl")
    for jsonl_file in all_jsonl_files:
        path = f"./all_jsonl/{jsonl_file}"
        with open (path, "r") as file:
            for row in file:
                data = json.loads(row)
                action_no = data["cases_act"]
                all_action_nos.append(action_no)
    return all_action_nos


def build_case_query_engine(action_no):
    pinecone.init(
        api_key=pinecone_api_key,
        environment=pinecone_environment
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
    print("3- Pinecone canvas already exists. Now we're connected.")

    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    filters = MetadataFilters(filters=[
        ExactMatchFilter(
            key = "cases_act",
            value = action_no
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

    return response.response_gen



def get_metadata(action_no):
    all_jsonl_files = os.listdir("./all_jsonl")
    for jsonl_file in all_jsonl_files:
        path = f"./all_jsonl/{jsonl_file}"
        with open (path, "r") as file:
            for row in file:
                data = json.loads(row)
                if action_no == data['cases_act']:
                    case_title = data['cases_title']

                    date_object = datetime.fromisoformat(data['date'])
                    case_date = date_object.strftime("%d %b, %Y")

                    case_db = data['db']
                    case_neutral_cit = data['neutral_cit']

                    parts = data['raw_case_num'].split("_")
                    case_link = f"https://www.hklii.hk/en/cases/{parts[1].lower()}/{parts[0]}/{parts[2]}"
    return case_title, case_date, case_db, case_neutral_cit, case_link


def add_logo():
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                background-image: url('https://i.imgur.com/XlV61vK.png');
                background-repeat: no-repeat;
                padding-top: 30px;
                background-position: 20px 50px;
                background-size: 300px 50px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

add_logo()


def main():

    st.title(":robot_face: Chat with Any Case")

    
    all_act_nos = get_all_case_nos()

    if 'show_iframe_cwc' not in st.session_state:
        st.session_state['show_iframe_cwc'] = False

    
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": f"Ask me a question about this case!"}
        ]

    if "streaming" not in st.session_state:
        st.session_state.streaming = False

        
    if "case_to_chat" not in st.session_state:
        st.session_state.case_to_chat = all_act_nos[0]


    case_title, case_date, case_db, case_neutral_cit, case_link = get_metadata(st.session_state['case_to_chat'])

    with st.sidebar:
        
        if st.session_state.case_to_chat != None:
            index = all_act_nos.index(st.session_state.case_to_chat)
        else:
            index = 0

        if st.session_state.streaming:
            st.selectbox('Which case would you like to chat with?', ("Please wait...",), disabled=True)
        else:
            selected_case_act_no = st.selectbox('Which case would you like to chat with?',
                                            (all_act_nos), 
                                            index=index)
        
        if selected_case_act_no != st.session_state.case_to_chat:
            st.session_state.case_to_chat = selected_case_act_no
            st.session_state.messages = [
                {"role": "assistant", "content": f"Ask me a question about case {st.session_state.case_to_chat}."}
            ]
        st.session_state['show_iframe_cwc'] = st.toggle('Show the case in chat window.', key=f"toggle_cwc", value=False)

        st.info("Please re-toggle the button when you change the case number.")

        html_content = f"""
            <style>
                .info-table {{
                    width: 100%;
                    border-collapse: collapse;
                }}
                .info-table th, .info-table td {{
                    border: 1px solid #dddddd;
                    text-align: left;
                    padding: 8px;
                }}
                .info-table tr:nth-child{{
                    background-color: rgba(255, 18, 32, 0.3);
                }}
                .info-table th {{
                    background-color: rgba(255, 18, 32, 0.3); /* Transparent orange background */
                    color: black;
                    font-weight: bold;
                }}
            </style>

            <table class="info-table">
                <tr>
                    <th>Date:</th>
                    <td>{case_date}</td>
                </tr>
                <tr>
                    <th>Action No.:</th>
                    <td>{st.session_state["case_to_chat"]}</td>
                </tr>
                <tr>
                    <th>Neutral Cit.:</th>
                    <td>{case_neutral_cit}</td>
                </tr>
                <tr>
                    <th>Title:</th>
                    <td>{case_title}</td>
                </tr>
                <tr>
                    <th>Court:</th>
                    <td>{case_db}</td>
                </tr>
            </table>
        """
        st.markdown(html_content, unsafe_allow_html=True)
        
        # st.write(st.session_state['show_iframe_cwc'])
        # st.write(st.session_state['case_to_chat'])

    
    if prompt := st.chat_input("Your question"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.streaming = True

    # if st.session_state.streaming:
    #     st.warning("Please wait for the current answer to complete.")
    #     st.selectbox(
    #     'Which case would you like to chat with?',
    #     ("Please wait...",), disabled=True)
    #     case_to_chat = st.session_state.case_to_chat
    # else:
    #     if st.session_state.case_to_chat != None:
    #         index = all_act_nos.index(st.session_state.case_to_chat)
    #     else:
    #         index = 0
        
        # st.session_state.selected_case_to_chat = st.selectbox('Which case would you like to chat with?',
        #                                     (all_act_nos), 
        #                                     index=index)
        
        # if selected_case_act_no != st.session_state.case_to_chat:
        #     st.session_state.case_to_chat = selected_case_act_no
        #     st.session_state.messages = [
        #         {"role": "assistant", "content": f"Ask me a question about case {st.session_state.case_to_chat}."}
        #     ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    if st.session_state.show_iframe_cwc:
        st.components.v1.iframe(src=case_link, width=None, height=500, scrolling=True)

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Generating answer..."):
                response = query_case(st.session_state.case_to_chat, prompt)
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