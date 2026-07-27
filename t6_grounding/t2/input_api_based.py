from enum import StrEnum
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field

from commons.constants import OPENAI_API_KEY
from t6_grounding.user_service_client import UserServiceClient


QUERY_ANALYSIS_PROMPT = """
You act as a query analysis system.

## INSTRUCTIONS
- Analyze the user question and extract explicit search values
- Available search fields: name, surname, email
- Map extracted values to the appropriate search fields
- Only extract values that are clearly stated - do not infer or assume

## EXAMPLES
"Who is John?" → name: "John",
"Find John Smith" → name: "John", surname: "Smith"
"""

SYSTEM_PROMPT = """
You a RAG-powered assistant

## STRUCTURE OF USER'S MESSAGE:
`CONTEXT` - The retrieved user data formatted as text.
`QUERY` - The user's original question.

## INSTRUCTIONS
- Read the RAG CONTEXT carefully before answering.
- Answer the USER QUESTION using ONLY the provided RAG CONTEXT and conversation history.
- If no relevant information exists in RAG CONTEXT, respond with: "The question cannot be answered using the provided context."
- If presenting user information, format it clearly and organize it in a readable manner.
"""

USER_PROMPT = """
## RAG CONTEXT:
{context}

## USER QUESTION"
{question}
"""


class SearchField(StrEnum):
    NAME = "name"
    SURNAME = "surname"
    EMAIL = "email"


class SearchRequest(BaseModel):
    search_field: SearchField = Field(description="Search field")
    search_value: str = Field(description="Search value. Sample: Adam.")


class SearchRequests(BaseModel):
    search_request_parameters: list[SearchRequest] = Field(
        description="List of search parameters to execute",
        default_factory=list
    )


llm_client = OpenAI(api_key=OPENAI_API_KEY)

user_client = UserServiceClient()


def retrieve_context(user_question: str) -> list[dict[str, Any]]:
    """Retrieve context from user question."""
    messages = [
        {
            "role": "system",
            "content": QUERY_ANALYSIS_PROMPT
        },
        {
            "role": "user",
            "content": user_question,
        },
    ]

    response = llm_client.beta.chat.completions.parse(
        model="gpt-4.1-nano",
        messages=messages,
        temperature=0.0,
        response_format=SearchRequests
    )
    parsed = response.choices[0].message.parsed
    if not parsed.model_dump().get('search_request_parameters'):
        print("No specific search parameters found!")
        return []

    search_params_dict = {param.search_field.value: param.search_value for param in parsed.search_request_parameters}

    print(f"Searching with parameters: {search_params_dict}")

    return user_client.search_users(**search_params_dict)


def augment_prompt(user_question: str, context: list[dict[str, Any]]) -> str:
    """Create augmented prompt from user question and context."""
    prompt_str = ""
    for param in context:
        prompt_str += f"User:\n"
        for key, value in param.items():
            prompt_str += f"  {key}: {value}\n"
        prompt_str += "\n"

    augmented_prompt = USER_PROMPT.format(context=prompt_str, question=user_question)
    print(f"Augmented prompt: {augmented_prompt}")

    return augmented_prompt


def generate_answer(augmented_prompt: str) -> str:
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

    response = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.0
    )

    content = response.choices[0].message.content
    if not content:
        return ""

    return content


def main():
    print("Query samples:")
    print(" - I need user emails that filled with hiking and psychology")
    print(" - Who is John?")
    print(" - Find users with surname Adams")
    print(" - Do we have smbd with name John that love painting?")

    while True:
        user_question = input("> ").strip()
        if user_question:
            if user_question.lower() in ['quit', 'exit']:
                break

            print("\n--- Retrieving context ---")
            context = retrieve_context(user_question)

            if context:
                print("\n--- Augmenting prompt ---")
                augmented_prompt = augment_prompt(user_question, context)

                print("\n--- Generating answer ---")
                answer = generate_answer(augmented_prompt)
                print(f"\nAnswer: {answer}\n")

            else:
                print("\n--- No relevant information found ---")


if __name__ == "__main__":
    main()


# The problems with API based Grounding approach are:
#   - We need a Pre-Step to figure out what field should be used for search (Takes time)
#   - Values for search should be correct (✅ John -> ❌ Jonh)
#   - Is not so flexible
# Benefits are:
#   - We fetch actual data (new users added and deleted every 5 minutes)
#   - Costs reduce