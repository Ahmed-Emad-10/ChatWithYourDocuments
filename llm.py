'''
SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using only the context provided. "
    "If the context is empty or does not contain the answer, say you don't have that information."
)


def build_llm(temperature: float):
    """Return a LangChain chat model. Configure your provider here.

    Example (any provider supported by LangChain works, after installing its package
    and setting its API key in .env or the environment):

        from langchain.chat_models import init_chat_model
        return init_chat_model("<model-name>", model_provider="<provider>", temperature=temperature)
    """
    raise NotImplementedError("Configure your LLM provider in llm.py -> build_llm()")


async def generate_answer(question: str, context: str, temperature: float) -> str:
    llm = build_llm(temperature)
    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", f"Context:\n{context}\n\nQuestion: {question}"),
    ]
    response = await llm.ainvoke(messages)
    return response.content


'''
'''


from langchain_ollama import ChatOllama

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using only the context provided. "
    "If the context is empty or does not contain the answer, say you don't have that information."
)


def build_llm(temperature: float):
    return ChatOllama(
        model="gemma4:e4b",
        temperature=temperature,
        base_url="http://127.0.0.1:11434"
    )


async def generate_answer(question: str, context: str, temperature: float) -> str:
    llm = build_llm(temperature)

    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", f"Context:\n{context}\n\nQuestion: {question}"),
    ]

    response = await llm.ainvoke(messages)
    return response.content




'''

import os
from langchain_google_genai import ChatGoogleGenerativeAI

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using only the context provided. "
    "If the context is empty or does not contain the answer, say you don't have that information."
)


def build_llm(temperature: float):
    api_key = "AIzaSyASSyQHSutITvoTbdMFHnYrweR7OfAblBU"

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is not set. Add your Gemini API key to your environment."
        )

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=temperature,
        google_api_key=api_key,
    )


async def generate_answer(question: str, context: str, temperature: float) -> str:
    llm = build_llm(temperature)

    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", f"Context:\n{context}\n\nQuestion: {question}"),
    ]

    response = await llm.ainvoke(messages)
    return response.content