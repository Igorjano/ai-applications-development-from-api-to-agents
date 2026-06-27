from commons.constants import OPENAI_API_KEY, OPENAI_EMBEDDINGS_ENDPOINT, OPENAI_CHAT_COMPLETIONS_ENDPOINT
from commons.models.conversation import Conversation
from commons.models.message import Message
from commons.models.role import Role
from t5_rag_advanced.chat.chat_completion_client import ChatCompletionClient
from t5_rag_advanced.embeddings.embeddings_client import EmbeddingsClient
from t5_rag_advanced.embeddings.text_processor import TextProcessor, SearchMode


SYSTEM_PROMPT = """
You are RAG powered assistant to answer questions how to use Microwave Oven.

## Structure of User message:
`RAG CONTEXT` - Retrieved documents relevant to the query.
`USER QUESTION` - The user's actual question.

## Instructions:
- Use information from `RAG CONTEXT` as context when answering the `USER QUESTION`.
- Cite specific sources when using information from the context.
- Answer ONLY based on conversation history and RAG context.
- If no relevant information exists in `RAG CONTEXT` or conversation history, state that you cannot answer the question.
"""

USER_PROMPT = """
## RAG CONTEXT:
{context}

## USER QUESTION"
{question}
"""
embedding_client = EmbeddingsClient(endpoint=OPENAI_EMBEDDINGS_ENDPOINT, model_name='text-embedding-3-small', api_key=OPENAI_API_KEY)
chat_completion_client = ChatCompletionClient(endpoint=OPENAI_CHAT_COMPLETIONS_ENDPOINT, model_name='gpt-5.2', api_key=OPENAI_API_KEY)
processor = TextProcessor(embeddings_client=embedding_client, db_config={'host': 'localhost','port': 5433,'database': 'vectordb','user': 'postgres','password': 'postgres'})


def main():
    print('Creating vectors table with embeddings ...')
    processor.process_text_file(file_name='embeddings/microwave_manual.txt', truncate_table=True)

    conversation = Conversation()
    conversation.add_message(Message(Role.SYSTEM, content=SYSTEM_PROMPT))

    print('🤖 RAG-powered assistant is at your service. Ask your questions or type "exit" to quit.')
    while True:
        user_input = input("⌨️: ")
        if user_input.lower() == "exit":
            print("Exiting chat session.")
            break

        context = processor.search(search_mode=SearchMode.COSINE_DISTANCE, query=user_input)
        augmented_prompt = USER_PROMPT.format(question=user_input, context=context)

        conversation.add_message(Message(role=Role.USER, content=augmented_prompt))

        completion = chat_completion_client.get_completion(conversation.messages)
        print(f'✨: {completion.content}')

        conversation.add_message(completion)


main()
