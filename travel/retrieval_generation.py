from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from travel.ingest import ingestdata

def generation(vstore):
    retriever = vstore.as_retriever(search_kwargs={"k": 3})

    PRODUCT_BOT_TEMPLATE = """
    Your travel bot is an expert in travel recommendations and user queries.
    It analyzes location and reviews to provide accurate and helpful responses.
    Ensure your answers are relevant to the context and refrain from straying off-topic.
    Your responses should be concise and informative.
    If question is not related to context respond with the answer 'I apologize but i do not have answer for this question'

    CONTEXT:
    {context}

    QUESTION: {question}

    YOUR ANSWER:
    
    """


    prompt = ChatPromptTemplate.from_template(PRODUCT_BOT_TEMPLATE)

    llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k',temperature=0.1)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

if __name__=='__main__':
    vstore = ingestdata("done")
    chain  = generation(vstore)
    print(chain.invoke("can you tell me the best location to visit?"))