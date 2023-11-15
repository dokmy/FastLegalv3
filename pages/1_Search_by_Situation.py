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
from datetime import datetime

# load_dotenv("./.env")


try:
    # Check if we are running on Streamlit Cloud
    if 'OPENAI_API_KEY' in st.secrets:
        openai.api_key = st.secrets["OPENAI_API_KEY"]
        pinecone_api_key = st.secrets["PINECONE_API_KEY"]
        pinecone_environment = st.secrets["PINECONE_ENVIRONMENT"]
    else:
        raise KeyError("Running in local environment")
except (FileNotFoundError, KeyError):
    # Load from .env file for local development
    load_dotenv("./.env")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")


    


    

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
print("1- Pinecone canvas already exists. Now we're connected.")


def build_context(model_name):
    llm_predictor = LLMPredictor(
        llm=ChatOpenAI(temperature=0, model_name=model_name)
        )
    return ServiceContext.from_defaults(llm_predictor=llm_predictor)


def build_case_query_engine(case_act_no):
        
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    filters = MetadataFilters(filters=[
        ExactMatchFilter(
            key = "cases_act",
            value = case_act_no
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


@st.cache_resource
def query_case(case_act_no, query):
    query_engine = build_case_query_engine(case_act_no)
    response = query_engine.query(query)

    return response.response_gen


def get_embedding(query):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=query
    )
    return response['data'][0]['embedding']


def query_pinecone(query_embedding, filters):
    list_of_case_metadata = []

    index_name = "cases-index"
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            index_name,
            dimension=1536,
            metric='cosine'
        )
        print("Pinecone canvas does not exist. Just created and connected.")
    pinecone_index = pinecone.Index(index_name)
    print("1b - Pinecone canvas already exists. Now we're connected.")

    results = pinecone_index.query(
        vector=query_embedding,
        filter={"case_prefix": {"$in": filters}},
        top_k=50,
        include_metadata=True
    )

    matches = results['matches']
    for match in matches:
        node_content = json.loads(match['metadata']['_node_content'])
        case_metadata = node_content['metadata']
        list_of_case_metadata.append(case_metadata)
                                
    return list_of_case_metadata



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





st.title(":mag: Search by Situation")

st.sidebar.title("Search by Situation")

with st.sidebar:
    
    st.markdown("**Describe your client's situation in the following box.**")
    user_input = st.sidebar.text_area("Be as specific as possible:", placeholder="E.g. My client slips and falls in a shopping mall while working...")
    st.markdown("**Select types of cases to search:**")
    
    options1 = st.multiselect(
    'Court of Final Appeal',
    ['FACV','FACC','FAMV','FAMC','FAMP'],
    None)
    options2 = st.multiselect(
    'Court of Appeal',
    ['CACV','CACC','CAAR','CASJ','CAQL','CAAG','CAMP'],
    None)
    options3 = st.multiselect(
    'Court of First Instance - Civil Matters',
    ['HCA','HCAL','HCAJ','HCAD','HCB','HCCL','HCCW','HCSD','HCBI','HCCT','HCMC','HCMP','HCCM','HCPI','HCBD','HCBS','HCSN','HCCD','HCZZ','HCMH','HCIP'],
    None)
    options4 = st.multiselect(
    'Court of First Instance - Criminal & Appeal Cases',
    ['HCCC','HCMA','HCLA','HCIA','HCSA','HCME','HCOA','HCUA','HCED','HCAA','HCCP'],
    None)
    options5 = st.multiselect(
    'Court of First Instance - Probate Matters',
    ['HCAP','HCAG','HCCA','HCEA','HCRC','HCCI','HCCV','HCUA'],
    None)
    options6 = st.multiselect(
    'Competition Tribunal',
    ['CTAR','CTEA','CTA','CTMP'],
    None)
    options7 = st.multiselect(
    'District Court',
    ['DCCJ','DCCC','DCDT','DCTC','DCEC','DCEO','DCMA','DCMP','DCOA','DCPI','DCPA','DCSA','DCZZ','DCSN'],
    None)
    options8 = st.multiselect(
    'Family Court',
    ['FCMC','FCJA','FCMP','FCAD','FCRE'],
    None)
    options9 = st.multiselect(
    'Lands Tribunal',
    ['LDPA','LDPB','LDPD','LDPE','LDRT','LDNT','LDLA','LDRA','LDBG','LDGA','LDLR','LDHA','LDBM','LDDB','LDDA','LDMT','LDCS','LDRW','LDMR','LDMP'],
    None)
    options10 = st.multiselect(
    'Others',
    ['CCDI','ESCC','ESS','FLCC','FLS','KCCC','KCS','KTCC','KTS','LBTC','OATD','STCC','STMP','STS','SCTC','TMCC','TMS','WKCC','WKS'],
    None)

    filters = options1 + options2 + options3 + options4 + options5 + options6 + options7 + options8 + options9 + options10
    # st.write(filters)
    submit_button = st.sidebar.button("Search")
    

if "search_results" not in st.session_state:
    st.session_state.search_results = None


if submit_button:

    with st.spinner('Generating answers...'):
        
        st.markdown("Searching for cases")
    
        query = user_input
        query_embedding = get_embedding(query)
        list_of_case_metadata = query_pinecone(query_embedding, filters)
        # st.write(list_of_case_metadata)
        final_list_of_case_metadata = list_of_case_metadata[:5]
        st.session_state.search_results = final_list_of_case_metadata
        st.markdown(f"**Found {len(list_of_case_metadata)} case(s). Showing top {len(final_list_of_case_metadata)} case(s) below with explanation:**")
        

        i = 0
        for case in st.session_state.search_results:
            i = i+1
            with st.expander(f"Case {i}: {case['cases_act']}"):
                # button = st.button("Chat with this case!", key=f"{case_act_no}")
                chat_link = f'[Chat with this case!](http://localhost:8999/?case_act_no=changelater)'
                raw_date = datetime.fromisoformat(case["date"])
                formatted_date = raw_date.strftime("%d %b, %Y")
                year, court, case_number = case['raw_case_num'].split("_")
                link = f"https://www.hklii.hk/en/cases/{court.lower()}/{year}/{case_number}"
                ans_box = st.empty()

                # stream = []
                if case["cases_act"] in st.session_state:
                    stream = st.session_state[f"{case['cases_act']}"]
                else:
                    stream = []

                for res in query_case(case['cases_act'], query):
                    stream.append(res)
                    answer = "".join(stream).strip()
                    ans_box.markdown(
                        f'<h3>{case["cases_title"]}</h3>'
                        '<ul>'
                        f'<li>Date: {formatted_date}</li>'                        
                        f'<li>Neutral Citation: {case["neutral_cit"]}</li>'
                        f'<li><a href="{link}" target="_blank">Click here to the case</a></li>'
                        '</ul><br>'
                        f'{answer}</div>', 
                        unsafe_allow_html=True
                                    )
                    st.session_state[f"{case['cases_act']}"] = stream

    
