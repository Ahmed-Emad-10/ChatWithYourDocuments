import os
from langchain_google_genai import ChatGoogleGenerativeAI

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using only the context provided. "
    "If the context is empty or does not contain the answer, say you don't have that information."
)


def build_llm(temperature: float):
    api_key = os.getenv("GOOGLE_API_KEY")

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