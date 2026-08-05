import asyncio
from typing import Any, Optional
import json

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from pydantic import BaseModel, Field

from commons.constants import OPENAI_API_KEY
from t6_grounding.user_service_client import UserServiceClient


# HOBBIES SEARCHING WIZARD
# Searches users by hobbies and provides their full info in JSON format:
#   Input: `I need people who love to go to mountains`
#   Output:
#     ```json
#       "rock climbing": [{full user info JSON},...],
#       "hiking": [{full user info JSON},...],
#       "camping": [{full user info JSON},...]
#     ```

SYSTEM_PROMPT = """
You a RAG-powered searching engine that assists search users by hobbies and provides Named Entity Extraction in Json format.

## STRUCTURE OF USER'S MESSAGE:
`RAG CONTEXT` - The retrieved user data formatted as text.
`USER QUESTION` - The user's original question.

## INSTRUCTIONS
- Read the RAG CONTEXT carefully before answering.
- Answer the USER QUESTION using ONLY the provided RAG CONTEXT and conversation history.
- If no relevant information exists in RAG CONTEXT, respond with: "The question cannot be answered using the provided context."
"""

USER_PROMPT =  """
## RAG CONTEXT:
{context}

## USER QUESTION"
{query}
"""


def format_user_document(user: dict[str, Any]) -> str:
    return f"User:\n  id: {user.get('id')}\n  about me: {user.get('about_me')}\n"


class GroupingIds(BaseModel):
    hobby: str = Field(description="Hobby name")
    ids: list[int] = Field(description="List of users id which contains hobby requested by user")


class GroupingHobbies(BaseModel):
    hobbies: list[GroupingIds] = Field(description="List of hobbies with users id")


class InputVectorGrounder:
    def __init__(self, embeddings: OpenAIEmbeddings, llm_client: OpenAI):
        self.embeddings = embeddings
        self.llm_client = llm_client
        self.user_client = UserServiceClient()
        self.vectorstore = None

    async def __aenter__(self):
        await self.initialize_vectorstore()
        return self

    async def initialize_vectorstore(self, batch_size: int = 100):
        """Initialize vector store with all current users."""
        print("🔎 Loading all users...")
        users = self.user_client.get_all_users()
        documents = [Document(page_content=format_user_document(user)) for user in users]
        batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]

        ids = [str(user.get("id")) for user in users]
        ids_batches = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]

        print(f"↗️ Initialize vectorstore for {len(documents)} documents...")
        self.vectorstore = Chroma(
            collection_name="users_collection",
            embedding_function=self.embeddings,
        )

        coroutines = [self.vectorstore.aadd_documents(documents=batch, ids=batch_id) for batch, batch_id in zip(batches, ids_batches)]
        await asyncio.gather(*coroutines)
        print("✅ Vectorstore is initialized.")


    async def _update_vectorstore(self):
        print("🔄 Updating vectorstore...")
        users = self.user_client.get_all_users()
        vectorstore_data = self.vectorstore.get()

        ids = set(str(user.get("id")) for user in users)
        vectorstore_ids = set(str(user_id) for user_id in vectorstore_data.get("ids", []))

        added_ids = ids - vectorstore_ids
        deleted_ids = vectorstore_ids - ids

        new_users = [user for user in users if str(user["id"]) in added_ids]
        new_documents = [Document(page_content=format_user_document(user)) for user in new_users]
        new_ids = [str(user["id"]) for user in new_users]

        if deleted_ids:
            await self.vectorstore.adelete(ids=list(deleted_ids))
            print(f"❌ {len(deleted_ids)} users have been removed.")

        if new_documents:
            await self.vectorstore.aadd_documents(documents=new_documents, ids=new_ids)
            print(f"❇️ {len(new_users)} users have been added.")

    async def retrieve_context(self, query: str, k: int = 10, score: float = 0.1) -> str:
        """Retrieve the context of the given query with updated vectorstore."""
        await self._update_vectorstore()
        print("Retrieving context...")
        context_score = self.vectorstore.similarity_search_with_relevance_scores(query, k=k, score_threshold=score)

        context_parts = []
        for doc, relevance_score in context_score:
            context_parts.append(doc.page_content)
            print(f"Retrieved (Score: {relevance_score:.3f}): {doc.page_content}")
            print("=" * 100 + "\n")

        return "\n\n".join(context_parts)

    @staticmethod
    def augment_prompt(query: str, context: str) -> str:
        return USER_PROMPT.format(query=query, context=context)

    def generate_answer(self, augmented_prompt: str) -> GroupingHobbies:
        """Generate answer based on augmented prompt."""
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": augmented_prompt,
            },
        ]

        response = self.llm_client.beta.chat.completions.parse(
            model="gpt-4.1-nano",
            messages=messages,
            temperature=0.0,
            response_format=GroupingHobbies
        )

        parsed = response.choices[0].message.parsed

        return parsed

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

class OutputAPIGrounder:
    def __init__(self):
        self.user_client = UserServiceClient()

    async def ground_response(self, parsed_results: GroupingHobbies):
        for result in parsed_results.hobbies:
            print(f"Hobby: {result.hobby}")
            print(f"Users: {await self._get_users_data(result.ids)}")
            print("-" * 100)

    async def _get_users_data(self, ids: list[int]) -> list[dict[str, Any]]:
        async def get_existing_users(user_id):
            try:
                return await self.user_client.get_user(user_id)
            except Exception as e:
                if "404" in str(e):
                    return None
                raise e


        coroutines = [get_existing_users(id_)for id_ in ids]
        users_data = await asyncio.gather(*coroutines)

        return [user for user in users_data]


async def main():
    llm_client = OpenAI(api_key=OPENAI_API_KEY)
    output_grounder = OutputAPIGrounder()
    embeddings = OpenAIEmbeddings(
        model='text-embedding-3-small',
        api_key=OPENAI_API_KEY,
        dimensions=384,
    )

    async with InputVectorGrounder(embeddings, llm_client) as rag:
        print("Query samples:")
        print(" - I need user that filled with hiking and psychology")
        print(" - I need people who love to go to mountains")
        print(" - I need users who like fishing and camping")

        while True:
            user_question = input("> ").strip()
            if user_question.lower() in ['quit', 'exit']:
                break

            context = await rag.retrieve_context(user_question)
            if context:
                augmented_prompt = rag.augment_prompt(user_question, context)
                parsed_results = rag.generate_answer(augmented_prompt)
                await output_grounder.ground_response(parsed_results)


if __name__ == '__main__':
    asyncio.run(main())