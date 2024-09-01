from langchain_core.pydantic_v1 import BaseModel, Field
from typing_extensions import Annotated, TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

# This code was inspired by an example from the "web-explorer" project by LangChain AI.
# Source: https://github.com/langchain-ai/web-explorer/

class Claims(BaseModel):
    # List of claims
    claims: List[str] = Field(description="List of claims to be extracted from the text")

def extract_claims_chain(model):
    prompt = ChatPromptTemplate.from_template(
        """
        Extract atomic and unambiguous claims from the input text provided below.
        Make sure that:
        1. The claims should not contain ambiguous references such as 'he', 'she', 'it', etc.
        2. The claims should use the full name of entities (e.g., 'John Doe' instead of 'he').
        3. The claims should be in the form of complete, self-contained sentences.
        4. If the output is too lengthy, prioritize the most important claims without losing context.
        Here is the text to extract claims from: 
        "{text}"
        """
    )
    structured_llm = model.with_structured_output(Claims)
    chain = prompt | structured_llm
    return chain

# This gets expensive quickly
def setup_web_retriever(model):
    import faiss
    from langchain.vectorstores import FAISS
    from langchain_community.embeddings import OpenAIEmbeddings
    from langchain_community.docstore.in_memory import InMemoryDocstore
    from langchain_community.retrievers import WebResearchRetriever
    from langchain_google_community import GoogleSearchAPIWrapper

    embeddings_model = OpenAIEmbeddings()
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore_public = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

    # Set up Google Search API wrapper for web retrieval
    search = GoogleSearchAPIWrapper()

    # Set up web retriever using vector store and LLM model
    web_retriever = WebResearchRetriever.from_llm(
        vectorstore=vectorstore_public,
        llm=model,
        allow_dangerous_requests=True,
        search=search,
        num_search_results=3
    )
    return web_retriever

# Format documents for model input
def format_docs(docs):
    formatted_docs = []
    for doc in docs:
        content = doc.page_content
        source = doc.metadata.get('source', 'Unknown source')
        title = doc.metadata.get('title', 'Unknown title')
        formatted_docs.append(f"{content}\n(Source: {title} {source})")
    return "\n\n".join(formatted_docs)

# TypedDict for structured model output with sources and verdict
class AnswerWithSources(TypedDict):
    verdict: Annotated[str, ..., "Verdict based on the answer content, options are: True, False, or Uncertain"]
    answer: str
    sources: Annotated[List[str], ..., "List of sources ([number] author + web link) used to answer the question"]

# Function to create the fact-checking chain
def fact_check_chain(model):
    prompt_template = """
        You are an expert AI assistant. Your task is to provide accurate and concise answers based on the provided context. 
        **Important:** Ensure that every claim in your answer is backed by a specific source. Use in-text citations in the format [1], [2], etc., corresponding to the sources provided.

        After answering the question, list the sources you used in a separate "SOURCES" section. Each source should be listed in the following format:
        [1] Author, "Title," URL.

        **Guidelines:**
        1. **Answer Format**: Directly answer the question in a clear and concise manner.
        2. **Citations**: Every piece of information or claim in your answer must be accompanied by an in-text citation like [1], [2], etc.
        3. **Source Listing**: List only the sources you directly referenced in your answer. If no sources are provided or relevant, write "SOURCES: None."
        4. **Content Integrity**: If the provided summaries lack sufficient information to answer the question confidently, clearly state that the information is incomplete and cite accordingly.

        Format your response as follows:

        **Answer:** [Your answer to the question with in-text citations]
        **Verdict** [Provide verdict based on the information available in the context]
        **Sources:** [List of sources, in proper format. If none, write "None"]

        **context:**
        {context}

        **Question:**
        """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_template),
            ("human", "{input}"),
        ]
    )

    rag_chain_from_docs = (
        {
            "input": lambda x: x["input"],
            "context": lambda x: format_docs(x["context"]),
        }
        | prompt
        | model.with_structured_output(AnswerWithSources)
    )

    retriever = setup_web_retriever(model)
    retrieve_docs = (lambda x: x["input"]) | retriever

    chain = RunnablePassthrough.assign(context=retrieve_docs).assign(
        answer=rag_chain_from_docs
    )

    return chain

# Function to create the full processing chain
def full_chain(model):
    extract_chain = extract_claims_chain(model)
    fact_check = fact_check_chain(model)

    chain = (
        extract_chain  # Step 1: Extract claims from the input text
        | (lambda result: result)  # Extract the list of claims from the result
        | (  # Step 2: Run fact-checking on each claim
            lambda claims: [
                fact_check.invoke({"input": claim}) for claim in claims.claims
            ]
        )
        | (  # Step 3: Combine all fact-checked results
            lambda results: {
                "fact_checked_claims": results  # Return the combined results
            }
        )
    )

    return chain


# TODO: Implement parallel processing for fact-checking
def full_chain_parallel(model):
    extract_chain = extract_claims_chain(model)

    def run_fact_checks_in_parallel(claims):
        results = []
        with ThreadPoolExecutor() as executor:
            # Run fact-checking in parallel
            fact_check = fact_check_chain(model)
            future_to_claim = {executor.submit(fact_check.invoke, {"input": claim}): claim for claim in claims.claims}
            for future in as_completed(future_to_claim):
                claim = future_to_claim[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    # Handle exceptions that occur during fact-checking
                    results.append({"claim": claim, "error": str(e)})
        return results

    chain = (
        extract_chain  # Step 1: Extract claims from the input text
        | (lambda result: result)  # Extract the list of claims from the result
        | (  # Step 2: Run fact-checking on each claim in parallel
            run_fact_checks_in_parallel
        )
        | (  # Step 3: Combine all fact-checked results
            lambda results: {
                "fact_checked_claims": results  # Return the combined results
            }
        )
    )

    return chain









