import pinecone
import dotenv
from dotenv import load_dotenv
from llama_index import (
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
    VectorStoreIndex
)
from llama_index.retrievers import VectorIndexRetriever
from llama_index.vector_stores import PineconeVectorStore
from llama_index.vector_stores.types import ExactMatchFilter, MetadataFilters
import openai
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def count_cases_in_Pinecone():

    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENVIORONMENT")
    )

    index_name = "cases-index"

    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine"
        )
        print("Pinecone index doesn't exist. Just created.")
    pinecone_index = pinecone.Index(index_name)
    print("Pinecone index existed. Connected.")

    service_context = ServiceContext.from_defaults()

    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index,
        # metadata_filters={"content_type": "case_itself" }
        )
    
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        # metadata_filters={"content_type": "case_itself"}
    )

    #count case summaries
    cs_filters = MetadataFilters(
        filters=[
            ExactMatchFilter(
                key="content_type",
                value="case_summary"
            )
        ]
    )

    cs_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=10000,
        vector_store_query_mode="default",
        filters=cs_filters
    )

    query = ("My client slips in the mall.")

    nodes = cs_retriever.retrieve(query)
    print(f"Number of case_summaries nodes: {len(nodes)}")
    
    all_cs_case_num = []
    for node in nodes:
        case_num = node.metadata['case_number']
        if case_num not in all_cs_case_num:
            all_cs_case_num.append(case_num)

    print(f"Number of unique case_num in case_summaries: {len(all_cs_case_num)}")
    print(f"The case numberes are: {all_cs_case_num}")

    # Count case itself
    ci_filters = MetadataFilters(
        filters=[
            ExactMatchFilter(
                key="content_type",
                value="case_itself"
            )
        ]
    )

    ci_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=10000,
        vector_store_query_mode="default",
        filters=ci_filters
    )

    query = ("My client slips in the mall.")

    nodes = ci_retriever.retrieve(query)
    print(f"Number of case_itself nodes: {len(nodes)}")
    
    all_ci_case_num = []
    for node in nodes:
        case_num = node.metadata['case_number']
        if case_num not in all_ci_case_num:
            all_ci_case_num.append(case_num)

    print(f"Number of unique case_num in case_itself: {len(all_ci_case_num)}")
    print(f"The case numberes are: {all_ci_case_num}")


count_cases_in_Pinecone()

        