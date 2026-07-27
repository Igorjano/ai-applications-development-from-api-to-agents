import asyncio
from typing import Any

from openai import AsyncOpenAI

from commons.constants import OPENAI_API_KEY
from t6_grounding.user_service_client import UserServiceClient


BATCH_SYSTEM_PROMPT = """
You are a user search assistant.
Your task is to identify users that match the search criteria specified in the user's request.

## Structure of User message:
`CONTEXT` - The formatted user data.
`QUERY` - The user's search question.

## Instructions:
- Carefully analyze the user's search criteria and determine all relevant conditions.
- Examine every user record provided in the input.
- Compare each user against the search criteria.
- Return full details of matching users in their original format, without modifying, reformatting, summarizing, or omitting any fields.
- Preserve the original order of matching users.
- Do not add explanations, comments, reasoning, or any additional text.
- If no users match the search criteria, return exactly: 'NO_MATCHES_FOUND'.
"""

FINAL_SYSTEM_PROMPT = """
You are the final search results aggregator.
Your task is to compile the results returned from multiple batch searches into a single final response.

## Structure of User message:
`CONTEXT` - The formatted user data.
`QUERY` - The user's search question.

## Instructions:
- Review all batch search results.
- Combine all matching user records from the remaining batch results.
- Remove duplicate users while preserving their original information.
- Present the final list of unique matching users in a clear and organized format.
"""

USER_PROMPT =  """
## CONTEXT:
{context}

## USER QUERY"
{query}
"""

class TokenTracker:

    def __init__(self):
        self.total_tokens = 0
        self.batch_tokens = []

    def add_tokens(self, tokens: int):
        self.total_tokens += tokens
        self.batch_tokens.append(tokens)

    def get_summary(self) -> dict:
        summary = {
            "total_tokens": self.total_tokens,
            "batch_count": len(self.batch_tokens),
            "batch_tokens": self.batch_tokens
        }

        return summary


llm_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

token_tracker = TokenTracker()


def join_context(context: list[dict[str, Any]]) -> str:
    result = ''
    for user in context:
        result += "User:\n"
        for k, v in user.items():
            result += f" {k}: {v}\n"
        result += "\n"

    return result


async def generate_response(system_prompt: str, user_message: str) -> str:
    print("Processing...")

    messages = [
            {
                "role": "developer",
                "content": system_prompt
             },
            {
                "role": "user",
                "content": user_message,
            },
        ]

    response = await llm_client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=messages,
        temperature=0
    )

    total_tokens = response.usage.total_tokens if response.usage else 0
    token_tracker.add_tokens(total_tokens)

    content = response.choices[0].message.content

    print(f"CONTENT:\n{content}")
    print(f"TOKENS COUNT: {total_tokens}")

    return content


async def main():
    print("Query samples:")
    print(" - Do we have someone with name John that loves traveling?")

    user_question = input("> ").strip()
    if user_question:
        print("\n--- Searching user database ---")

        users = UserServiceClient().get_all_users()
        users_batches = [users[i:i + 100] for i in range(0, len(users), 100)]

        coroutines = [generate_response(BATCH_SYSTEM_PROMPT, USER_PROMPT.format(context=join_context(batch), query=user_question)) for batch in users_batches]
        batch_results = await asyncio.gather(*coroutines)

        print("\n--- Compiling results ---")
        if batch_results:
            relevant_results = [result for result in batch_results if result.strip() != "NO_MATCHES_FOUND"]

            print("\n=== SEARCH RESULTS ===")
            if relevant_results:
                combined_result = '\n\n'.join(relevant_results)
                await generate_response(FINAL_SYSTEM_PROMPT, combined_result)
            else:
                print("No results found.")
                print("Try to use different search criteria or keywords.")

    summary = token_tracker.get_summary()
    print("\n=== Performance ===")
    print(f'TOTAL API CALLS: {summary["batch_count"]}')
    print(f'TOTAL TOKENS: {summary["total_tokens"]}')


if __name__ == "__main__":
    asyncio.run(main())


# The problems with No Grounding approach are:
#   - If we load whole users as context in one request to LLM we will hit context window
#   - Huge token usage == Higher price per request
#   - Added + one chain in flow where original user data can be changed by LLM (before final generation)
# User Question -> Get all users -> ‼️parallel search of possible candidates‼️ -> probably changed original context -> final generation