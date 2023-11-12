import os
from dotenv import load_dotenv
import openai
from llama_index import(
    VectorStoreIndex,
    QuestionAnswerPrompt,
    LLMPredictor,
    ServiceContext
)
from langchain.chat_models import ChatOpenAI
import pinecone
from llama_index.vector_stores import PineconeVectorStore
from llama_index.vector_stores.types import ExactMatchFilter, MetadataFilters
import streamlit as st
import json
import requests
import datetime
import streamlit.components.v1 as components

load_dotenv("./.env")

try:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")

except:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    pinecone_api_key = st.secrets["PINECONE_API_KEY"]
    pinecone_environment = st.secrets["PINECONE_ENVIRONMENT"]

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
print("Pinecone canvas already exists. Now we're connected.")


def build_context(model_name):
    llm_predictor = LLMPredictor(
        llm=ChatOpenAI(temperature=0, model_name=model_name)
        )
    return ServiceContext.from_defaults(llm_predictor=llm_predictor)


def get_summary(act):
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    filters = MetadataFilters(filters=[
        ExactMatchFilter(
            key = "cases_act",
            value = act
        )
    ])

    PROMPT_TEMPLATE = (
        "Here are the context information:"
        "\n---------------------------------\n"
        "{context_str}"
        "\n---------------------------------\n"
        "You are a AI legal assistant for lawyers in Hong Kong. The context information is about a legal case. Please do this: {query_str}"
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

    query = "Please generate a concise summary of this case. Please don't generate more than 200 words."
    response = query_engine.query(query)

    return response.response_gen


def do_kw_search(cit,case_name, leg_name, parties,coram, representation, charge, all_of_these_words, any_of_these_words, exact_phrase, min_date, max_date):

    link = f"https://www.hklii.hk/api/advancedsearch?searchType=advanced&text={all_of_these_words}&anyword={any_of_these_words}&title={case_name}&citation={cit}&captitle={leg_name}&coram={coram}&charge={charge}&representation={representation}&parties={parties}&phrase={exact_phrase}&min_date={min_date}&max_date={max_date}&dbs=2,4,5,7,9,11,13,15,17,19,21,23,25"


    response = requests.get(link)

    if response.status_code == 200:
        data = response.json()
        try:
            result_count = data['count']
        except Exception as e:
            print(e)
            pass

        # results = data['results']
        st.session_state.kw_search_results = data['results']
        st.experimental_rerun()


def search_wrapper():
    cit = raw_cit.replace(" ", "+")
    case_name = raw_case_name.replace(" ", "+")
    leg_name = raw_leg_name.replace(" ", "+")
    parties = raw_parties.replace(" ", "+")
    coram = raw_coram.replace(" ", "+")
    representation = raw_representation.replace(" ", "+")
    charge = raw_charge.replace(" ", "+")
    all_of_these_words = raw_all_of_these_words.replace(" ", "+")
    any_of_these_words = raw_any_of_these_words.replace(" ", "+")
    exact_phrase = raw_exact_phrase.replace(" ", "+")

    raw_min_date = time_range[0]
    raw_max_date = time_range[1]
    formatted_raw_min_date_str = raw_min_date.strftime("%d/%m/%Y")
    formatted_raw_max_date_str = raw_max_date.strftime("%d/%m/%Y")
    min_date = formatted_raw_min_date_str.replace('/', '%2F')
    max_date = formatted_raw_max_date_str.replace('/', '%2F')

    do_kw_search(cit, case_name, leg_name, parties, coram, representation, charge, all_of_these_words, any_of_these_words, exact_phrase, min_date, max_date)


# st.set_page_config(
#     page_title="Search with keyword",
#     page_icon="🔍",
#     layout="wide",
#     initial_sidebar_state="expanded",
#     menu_items={
#         'Get Help': 'https://www.extremelycoolapp.com/help',
#         'Report a bug': "https://www.extremelycoolapp.com/bug",
#         'About': "# This is a header. This is an *extremely* cool app!"
#     }
# )


if "kw_search_results" not in st.session_state:
    st.session_state.kw_search_results = None
    
if "summaries" not in st.session_state:
    st.session_state.summaries = {}

if "searching" not in st.session_state:
    st.session_state.searching = False

# if "case_to_chat" not in st.session_state:
#     st.session_state.case_to_chat = None

st.title(":mag: Search by Keyword")
# st.write(st.session_state)
st.sidebar.title("Search by Keyword")


with st.sidebar:
    st.markdown("**Search in specific fields.**")
    raw_cit = st.text_input('Citation', placeholder='e.g.: [2002] HKCFI 1234')
    raw_case_name = st.text_input('Case Name', placeholder='e.g.: HKSAR v. CHAN KAM WAH;')
    raw_leg_name = st.text_input('Legislation name', placeholder='e.g.: JUSTICES OF THE PEACE ORDINANCE')
    raw_parties = st.text_input('Parties of Judgment', placeholder='e.g.: B & Q PLC')
    raw_coram = st.text_input('Coram of Judgment', placeholder='e.g.: E.C. Barnes')
    raw_representation = st.text_input('Parties representation', placeholder='e.g.: G. Alderdice')
    raw_charge = st.text_input('Charge', placeholder='e.g.: Dangerous driving')

    st.markdown("**Search in all fields.**")
    raw_all_of_these_words = st.text_input('All of these words', placeholder='e.g. breach fiduciary duty')
    raw_any_of_these_words = st.text_input('Any of these words', placeholder='e.g. waste pollution radiation')
    raw_exact_phrase = st.text_input('Exact phrase', placeholder='e.g. parliamentary sovereignty')

    st.markdown("**Search in all fields.**")
    today = datetime.datetime.now()
    five_years_ago = today - datetime.timedelta(days=25*365)
    today_date = today.date()
    five_years_ago_date = five_years_ago.date()
    time_range = st.date_input(
        'Pick a date range:', 
        (five_years_ago_date, today),
        min_value=datetime.date(1980, 1, 1),
        max_value=today_date,
        format="DD/MM/YYYY"
    )
    raw_min_date = time_range[0]
    raw_max_date = time_range[1]
    formatted_raw_min_date_str = raw_min_date.strftime("%d/%m/%Y")
    formatted_raw_max_date_str = raw_max_date.strftime("%d/%m/%Y")

    #Convert to search params
    cit = raw_cit.replace(" ", "+")
    case_name = raw_case_name.replace(" ", "+")
    leg_name = raw_leg_name.replace(" ", "+")
    parties = raw_parties.replace(" ", "+")
    coram = raw_coram.replace(" ", "+")
    representation = raw_representation.replace(" ", "+")
    charge = raw_charge.replace(" ", "+")
    all_of_these_words = raw_all_of_these_words.replace(" ", "+")
    any_of_these_words = raw_any_of_these_words.replace(" ", "+")
    exact_phrase = raw_exact_phrase.replace(" ", "+")
    min_date = formatted_raw_min_date_str.replace('/', '%2F')
    max_date = formatted_raw_max_date_str.replace('/', '%2F')

    submit_kw_search_button = st.sidebar.button("Search", key="submit_kw_search_button")

if submit_kw_search_button:
    with st.spinner("Searching..."):
        do_kw_search(cit,case_name, leg_name, parties,coram, representation, charge, all_of_these_words, any_of_these_words, exact_phrase, min_date, max_date)


if st.session_state.kw_search_results != None:
    if len(st.session_state.kw_search_results) > 50:
        result_shown = 50
    else:
        result_shown = len(st.session_state.kw_search_results)

    st.write(f"Found {len(st.session_state.kw_search_results)} results. Showing top {result_shown}.")

    i=1
    for result in st.session_state.kw_search_results[:result_shown]:
        with st.expander(f"Case {i}: {result['act']}"):
            ans_box = st.empty()
            act = result['act']
            raw_date_str = result['pub_date']
            date_obj = datetime.datetime.fromisoformat(raw_date_str)
            formatted_date = date_obj.strftime("%d %b, %Y")
            case_link = f"https://www.hklii.hk"+result["path"]
            ans_box.markdown(
                        f'<h3>{result["title"]}</h3>'
                        '<ul>'
                        f'<li>Court: {result["db"]}</li>'                        
                        f'<li>Neutral Citation: {result["neutral"]}</li>'
                        f'<li>Date: {formatted_date}</li>'  
                        # f'<li><a href="{case_link}" target="_blank">Click here to the case</a></li>'
                        # '</ul><br>',
                        # f'{answer}</div>', 
                        ,unsafe_allow_html=True
                                    )
            show_iframe = st.toggle('Show the case. Please allow several seconds to load.', key=f"toggle_{i}")
            if show_iframe:
                st.components.v1.iframe(src=case_link, width=None, height=500, scrolling=True)

            if act in st.session_state.summaries.keys():
                case_summary = st.session_state.summaries[str(act)]
                st.write("Case Summary: ")
                st.write(case_summary)
            else:
                if st.button("Let AI Summarise this case!", key=f"summarise_case_{act}_{i}"):
                    with st.spinner('Generating summary...'):
                        response = get_summary(act)
                        response_box = st.empty()
                        stream = []
                        for res in response:
                            stream.append(res)
                            answer = "".join(stream).strip()
                            response_box.markdown(answer)
                        st.session_state.summaries[str(act)] = answer
            
            i+=1




