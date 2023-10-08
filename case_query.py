import os
import json
import dotenv
import openai
import pinecone
from dotenv import load_dotenv
from llama_index import (
    SimpleDirectoryReader,
    ServiceContext,
    PineconeReader,
    GPTVectorStoreIndex,
    QuestionAnswerPrompt,
    LLMPredictor
)
from llama_index.vector_stores import PineconeVectorStore
from llama_index.storage.storage_context import StorageContext
from langchain.chat_models import ChatOpenAI

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def create_list_of_case_numbers():
    list_of_case_numbers = []
    cases_folder_path = "./data"
    cases = os.listdir(cases_folder_path)
    for case in cases:
        case_number = case.replace(".pdf","")
        list_of_case_numbers.append(case_number)
    return list_of_case_numbers


def build_docs(list_of_case_numbers):
    docs = {}
    for case_number in list_of_case_numbers:
        docs[case_number] = SimpleDirectoryReader(input_files=[f"./data/{case_number}.pdf"]).load_data()
    return docs

def build_context(model_name):
    llm_predictor = LLMPredictor(
        llm=ChatOpenAI(temperature=0, model_name=model_name)
        )
    return ServiceContext.from_defaults(llm_predictor=llm_predictor)


def build_dict_of_case_index(list_of_case_numbers, docs):
    case_indicies = {}
    
    #1. init pinecone
    # print("hahah", os.getenv("PINECONE_API_KEY"))

    pinecone.init(
        api_key = os.getenv("PINECONE_API_KEY"),
        environment = os.getenv("PINECONE_ENVIRONMENT")
    )

    #2. create the canvas and point to that canvas in Pinecone
    index_name = "cases-index"
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            index_name,
            dimension=1536,
            metric='cosine'
        )
        print("Pinecone canvas does not exist. Just created and connected.")
    pinecone_index = pinecone.Index(index_name)
    print("Pinecone canvas connected.")

    print(pinecone_index.fetch(ids=['DCPI003618_2019']))

    service_context = build_context("gpt-3.5-turbo")

    #3. create an instance of pinecone vector store
    for case_number in list_of_case_numbers:
        vector_store = PineconeVectorStore(
            pinecone_index=pinecone_index,
            metadata_filters={"case_number": case_number}
        )
        #4. create storage context to tell that the real vector store (later created in 5) that this Pinecone store will be stored in 3
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        #5. create the real vector index (GPTVectorStoreIndex) and store in 4
        case_indicies[case_number] = GPTVectorStoreIndex.from_documents(
            docs[case_number],
            storage_context=storage_context,
            service_context=service_context
        )
        case_indicies[case_number].index_struct.index_id = case_number
    print("Indexing complete.")
    return case_indicies

def test_pinecone_metadata():
    pinecone.init(
        api_key = os.getenv("PINECONE_API_KEY"),
        environment = os.getenv("PINECONE_ENVIRONMENT")
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
    print("Pinecone canvas connected.")

    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    
    nodes = vector_store.query(
        vector = [0]*1536,
        filter = {
            "file_name": "DCPI003618_2019.pdf"
        },
        top_k=10,
        include_metadata = True
    )
    
    for node in nodes.matches:
        deserialized_node_content = json.loads(node.metadata['_node_content'])
        file_name = deserialized_node_content['metadata']['file_name']
        print(file_name)
    

test_pinecone_metadata()

def query_case(search_result, query):
    # list_of_case_numbers = create_list_of_case_numbers()
    # docs = build_docs(list_of_case_numbers)
    # dict_of_case_indicies = build_dict_of_case_index(list_of_case_numbers, docs)

    #Init pincone and connect with canvas
    pinecone.init(
        api_key = os.getenv("PINECONE_API_KEY"),
        environment = os.getenv("PINECONE_ENVIRONMENT")
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
    print("Pinecone canvas connected.")

    #Query and receive nodes
    nodes = pinecone_index.query(
        vector = [0]*1536,
        filter = {
            "file_name": f"{search_result}.pdf"
        },
        top_k=5,
        include_metadata = True
    )

    #Just to test if the retrieval is successful and only searches the chosen case
    for node in nodes.matches:
        deserialized_node_content = json.loads(node.metadata['_node_content'])
        file_name = deserialized_node_content['metadata']['file_name']
        print(file_name)


    #Synthesize answers
    PROMPT_TEMPLATE = (
        "Here are the context information:"
        "\n------------------------------\n"
        "{context_str}"
        "\n------------------------------\n"
        "You are a AI legal assistant for lawyers in Hong Kong. Answer the follwing question in three parts. First, explained what happened in the case for reference in the context. Second, explain how this case is relevant to the following question:{query_str}. Lastly, answer this question: :{query_str} \n"
    )

    QA_PROMPT = QuestionAnswerPrompt(PROMPT_TEMPLATE)
    # query_engine = dict_of_case_indicies[search_result].as_query_engine(text_qa_template=QA_PROMPT)
    # res1 = query_engine.query(query)



    

    




    
