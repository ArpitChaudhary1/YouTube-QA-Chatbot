from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
from vector_store import retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda,RunnableParallel,RunnablePassthrough


# to build the context
def format_docs(retrieved_docs):
    context = "\n\n".join(docs.page_content for docs in retrieved_docs)
    return context


# loading llm
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    temperature= 0.3
)
model = ChatHuggingFace(llm=llm)



# Creating prompt Template
prompt = PromptTemplate(
    template="""
    You are a Helpful assistance.
    Answer ONLY from the provided transcript window.
    If the context window is insufficient, just say I don't know.
    
    {context}
    Question: {question} 
    """,
    input_variables=['context','question']
)

parser = StrOutputParser()

# Chains


parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

final_chain = parallel_chain | prompt | model | parser

result = final_chain.invoke('who is Demis and what he told about himself when he was young?')


print(result)