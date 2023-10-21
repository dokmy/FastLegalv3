import os
import json
import dotenv
import openai
import pinecone
import time
from dotenv import load_dotenv
from llama_index import (
    SimpleDirectoryReader,
    ServiceContext,
    PineconeReader,
    GPTVectorStoreIndex,
    QuestionAnswerPrompt,
    LLMPredictor,
    VectorStoreIndex
)
from llama_index.vector_stores import PineconeVectorStore
from llama_index.storage.storage_context import StorageContext
from langchain.chat_models import ChatOpenAI
from llama_index.vector_stores.types import ExactMatchFilter, MetadataFilters

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
        for doc in docs[case_number]:
            doc.metadata = {"content_type":"case_itself"}
            doc.metadata["case_number"] = case_number
    # print("Here are the number of LODOs: ",len(docs))
    # first_key, first_value = next(iter(docs.items()))
    # print("Here is the metadat of the first DO of the first LODO: ", docs[first_key][0].metadata)
    return docs


def build_context(model_name):
    llm_predictor = LLMPredictor(
        llm=ChatOpenAI(temperature=0, model_name=model_name)
        )
    return ServiceContext.from_defaults(llm_predictor=llm_predictor)


def build_dict_of_case_index(list_of_case_numbers, docs):
    case_indicies = {}
    
    #1. init pinecone
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
    print("Pinecone canvas already exists. Now we're connected.")

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
        # case_indicies[case_number].index_struct.index_id = case_number
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

    
    nodes = pinecone_index.query(
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
    

def query_case(dedup_search_results:list, query):

    print(f"4. Start query_search_engine. Time: {time.ctime(time.time())}")

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
    print("Pinecone canvas already exists. Now we're connected.")

    #Construct vector store from Pinecone
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    #Create a VectorStoreIndex from the existing vector store in Pinecone and then query it
    vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    vector_index_done_time = time.time()
    print(f"5. Second vector index done. Time: {time.ctime(time.time())}")

    #For each case, query with metadata and QAPrompt to return an answer
    final_answers = {}
    for case_num in dedup_search_results:

        #Define filters
        filters = MetadataFilters(filters=[
        ExactMatchFilter(
            key = "content_type",
            value = "case_summary"
        ),
        ExactMatchFilter(
            key = "case_number",
            value = case_num
        )]
    )
        
        #Define QAPrompt
        PROMPT_TEMPLATE = (
        "Here are the context information:"
        "\n------------------------------\n"
        "{context_str}"
        "\n------------------------------\n"
        "You are a AI legal assistant for lawyers in Hong Kong. Answer the follwing question in two parts. Break down these two parts with sub-headings. First, explained what happened in the case for reference in the context. Second, explain how this case is relevant to the following siutation or question: {query_str}. \n"
        )

        QA_PROMPT = QuestionAnswerPrompt(PROMPT_TEMPLATE)

        #Define query_engine
        query_engine = vector_index.as_query_engine(
            similarity_top_k = 3,
            vector_store_query_mode = "default",
            filters = filters,
            text_qa_template=QA_PROMPT,
            # streaming = True    
        )

        print(f"6a. In a for loop... Query Engine done. Time: {time.ctime(time.time())}")
        print("\n")
        #Peform query and return answers3
        response = query_engine.query(query)
        response.print_response_stream()
        print(response)
        print("\n")
        print(f"6b. Still in a for loop... A response is generated. Time: {time.ctime(time.time())}")
        print("\n")
        final_answers[str(case_num)] = str(response)
    
    print(f"7. Out of the for loop... All answers are created and saved to dict. Time: {time.ctime(time.time())}")
    
    return final_answers


# list_of_case_numbers = create_list_of_case_numbers()
# docs = build_docs(list_of_case_numbers)
# build_dict_of_case_index(list_of_case_numbers, docs)


    
