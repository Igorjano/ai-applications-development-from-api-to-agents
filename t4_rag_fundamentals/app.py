import os

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr

from commons.constants import OPENAI_API_KEY


_SYSTEM_PROMPT = """You are knowledge assistant to answer questions how to use Microwave Oven.

## Structure of User message:
`RAG CONTEXT` - Retrieved documents relevant to the query.
`USER QUESTION` - The user's actual question.

## Instructions:
- Use information from `RAG CONTEXT` as context when answering the `USER QUESTION`.
- Cite specific sources when using information from the context.
- Answer ONLY based on conversation history and RAG context.
- If no relevant information exists in `RAG CONTEXT` or conversation history, state that you cannot answer the question.
"""

_USER_PROMPT = """##RAG CONTEXT:
{context}


##USER QUESTION:
{query}"""


class MicrowaveRAG:

    def __init__(self, embeddings: OpenAIEmbeddings, llm_client: ChatOpenAI):
        self.llm_client = llm_client
        self.embeddings = embeddings
        self.vectorstore = self._setup_vectorstore()

    def _setup_vectorstore(self) -> VectorStore:
        """
        Load existing FAISS index from disk or create a new one.
        Returns:
              VectorStore: Initialized FAISS vectorstore.
        """
        print("Setup FAISS vectorstore...")

        if os.path.isdir('microwave_faiss_index'):
            print('Loading existing FAISS index from disk...')
            return FAISS.load_local("microwave_faiss_index", self.embeddings, allow_dangerous_deserialization=True)

        print('Creating new FAISS index...')
        return self._create_new_index()

    def _create_new_index(self) -> VectorStore:
        """
        Load the manual, split into chunks, embed, and save a new FAISS index.
        Returns:
              VectorStore: Newly created and saved FAISS vectorstore.
        """
        loader = TextLoader("microwave_manual.txt", encoding="utf-8")
        raw_documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            separators=["\n\n", "\n", "."]
        )

        documents = text_splitter.split_documents(raw_documents)
        vectorstore = FAISS.from_documents(documents, self.embeddings)
        vectorstore.save_local("microwave_faiss_index")

        return vectorstore

    def retrieve_context(self, query: str, k: int = 4, score=0.3):
        """
        Retrieve the context for a given query.
        Args:
              query (str): The query to retrieve the context for.
              k (int): The number of relevant documents(chunks) to retrieve.
              score (float): The similarity score between documents and query. Range 0.0 to 1.0.
        """
        chunks = []

        results =  self.vectorstore.similarity_search_with_relevance_scores(
            query=query,
            k=k,
            score_threshold=score
        )

        for doc, score in results:
            chunks.append(doc.page_content)
            print(f'Relevant score: {score}')

        return '\n\n'.join(chunk for chunk in chunks)

    def augment_prompt(self, query: str, context: str):
        """
        Inject retrieved context and user query into the prompt template.
        Args:
              query (str): The user's question.
              context (str): Retrieved context from the vectorstore.
        Returns:
              str: Formatted prompt ready for the LLM.
        """
        augmented_prompt = _USER_PROMPT.format(query=query, context=context)

        return augmented_prompt

    def generate_answer(self, augmented_prompt: str):
        """
        Send the augmented prompt to the LLM and return its response.
        Args:
              augmented_prompt (str): The prompt with injected context and query.
        Returns:
              str: The LLM-generated answer.
        """
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=augmented_prompt)
        ]

        response = self.llm_client.invoke(messages)
        print(f'✨: {response.content}')
        return response.content


def main(rag: MicrowaveRAG):
    print('🤖 RAG-powered assistant is at your service. Ask your questions or type "exit" to quit.')
    while True:
        user_input = input("⌨️: ")
        if user_input.lower() == "exit":
            print("Exiting chat session.")
            break

        context = rag.retrieve_context(user_input)
        augmented_prompt = rag.augment_prompt(user_input, context)
        rag.generate_answer(augmented_prompt)


embeddings = OpenAIEmbeddings(model='text-embedding-3-small', api_key=SecretStr(OPENAI_API_KEY))
client = ChatOpenAI(temperature=0.0, model='gpt-5.2', api_key=SecretStr(OPENAI_API_KEY))
main(MicrowaveRAG(embeddings, client))