import cassio
import os
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.graph_vectorstores import CassandraGraphVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.graph_vectorstores.extractors import (
    LinkExtractorTransformer,
    KeybertLinkExtractor,
)
from colorama import Style, init

from config import (
    logger,
    openai_api_key,
    astra_db_id,
    astra_token,
)
from langchain_utils import find_and_log_links, use_as_document_extractor
from utils import format_docs, ANSWER_PROMPT

# Initialize colorama
init(autoreset=True)

# Initialize embeddings and LLM using OpenAI
embeddings = OpenAIEmbeddings(api_key=openai_api_key)
llm = ChatOpenAI(temperature=1, model_name="gpt-4o-mini")

# Initialize Astra connection using Cassio
cassio.init(database_id=astra_db_id, token=astra_token)

graph_vector_store = CassandraGraphVectorStore(embeddings,node_table="demo")

class ChainManager:
    def __init__(self):
        self.similarity_chain = None
        self.traversal_chain = None

    def setup_chains(self):
        # Set up retrievers
        similarity_retriever = graph_vector_store.as_retriever(
            search_type='similarity',
            search_kwargs={
                "k": 10, 
                "depth": 1
            })
        traversal_retriever = graph_vector_store.as_retriever(
            search_type="traversal", search_kwargs={
                "k": 10, 
                "depth": 1,
                "score_threshold": 0.2,
            })

        # Set up chains
        self.similarity_chain = (
            {"context": similarity_retriever | format_docs, "question": RunnablePassthrough()}
            | ChatPromptTemplate.from_messages([ANSWER_PROMPT])
            | llm
        )
        self.traversal_chain = (
            {"context": traversal_retriever | format_docs, "question": RunnablePassthrough()}
            | ChatPromptTemplate.from_messages([ANSWER_PROMPT])
            | llm
        )

def main():
    try:
        #pages = PyPDFLoader("/Users/harshagubbichandrashekar/Desktop/demo-project/graphrag/docs/Football.pdf").load()
        directory_path = r"/Users/harshagubbichandrashekar/Desktop/demo-project/graphrag/docs/QPR"
        pdf_files = []
        for root, dirs, files in os.walk(directory_path):
                for file in files:
                    if file.lower().endswith('.pdf'):
                        pdf_files.append(os.path.join(root, file))
            

        for pdf_file in pdf_files:
            print(pdf_file)
            pages = PyPDFLoader(pdf_file).load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100,
                length_function=len,
                is_separator_regex=False,
            )
            pipeline = LinkExtractorTransformer([KeybertLinkExtractor()])
            pages = text_splitter.split_documents(pages)
            raw_documents = pipeline.transform_documents(pages)

            use_as_document_extractor(raw_documents)
            find_and_log_links(raw_documents)

            documents = text_splitter.split_documents(raw_documents)

            # Add documents to the graph vector store
            graph_vector_store.add_documents(documents)

    except Exception as e:
        logger.error("An error occurred: %s", e)


def compare_results(question):
    print(Style.BRIGHT + "\nQuestion:")
    print(Style.NORMAL + question)

    # Initialize ChainManager and set up chains
    chain_manager = ChainManager()
    chain_manager.setup_chains()
    
    output_answer = chain_manager.similarity_chain.invoke(question)
    print(Style.BRIGHT + "\n\nVector Similarity Result:")
    print(Style.NORMAL + output_answer.content)

    output_answer = chain_manager.traversal_chain.invoke(question)
    print(Style.BRIGHT + "\n\nGRAPH Traversal Result:")
    print(Style.NORMAL + output_answer.content)

async def get_similarity_result(chain_manager, question):
    """
    Gets the result from the similarity chain for a given question.
    
    Args:
        chain_manager (ChainManager): The chain manager instance.
        question (str): The question to be answered by the chain.
    """
    return chain_manager.similarity_chain.invoke(question).content


async def get_traversal_result(chain_manager, question):
    """
    Gets the result from the traversal chain for a given question.
    
    Args:
        chain_manager (ChainManager): The chain manager instance.
        question (str): The question to be answered by the chain.
    """
    return chain_manager.traversal_chain.invoke(question).content

if __name__ == "__main__":
    main()