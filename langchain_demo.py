from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings


def test_model():
    model = init_chat_model(model="mistral-small-latest")

    response = model.invoke("Why is the sky blue?")

    print(response.content)


class AgentCallback(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id = None, tags = None, metadata = None, **kwargs):
        print("CALLBACK:  LLM chain started: %s", prompts)

    def on_llm_end(self, response, *, run_id, parent_run_id = None, **kwargs):
        print("CALLBACK:  LLM chain ended: %s", response)


def test_basic_agent():
    def get_weather(city: str) -> str:
        """Get weather for a given city."""
        if city.lower().startswith("s"):
            return f"It's sunny in {city}!"
        else:
            return f"It's raining in {city}!"

    agent = create_agent(
        model="mistral-small-latest",
        tools=[get_weather],
        system_prompt="You are a helpful assistant",
        # debug=True,
    )

    config = {
        "callbacks": [AgentCallback()]
    }

    result = agent.invoke(
        input={"messages": [{"role": "user", "content": "hi, I'm Mark, what is the weather in sf and nyc?"}]},
        config=config
    )

    for message in result["messages"]:
        message.pretty_print()


def test_agent_memory():
    checkpointer = InMemorySaver()

    agent = create_agent(
        model="mistral-small-latest",
        system_prompt="You are a helpful assistant",
        checkpointer=checkpointer,
        middleware=[
            SummarizationMiddleware(
                model="mistral-small-latest",
                trigger=("messages", 8),
                keep=("messages", 4),
            )
        ]
    )

    config = {
        "configurable": {"thread_id": "1"},
        "callbacks": [AgentCallback()]
    }

    result = agent.invoke(
        input={"messages": [{"role": "user", "content": "hi, I'm Michael, how is life?"}]},
        config=config
    )

    result = agent.invoke(
        input={"messages": [{"role": "user", "content": "How much for a lightsabre? Just asking."}]},
        config=config
    )

    result = agent.invoke(
        input={"messages": [{"role": "user", "content": "Remember what is my name?"}]},
        config=config
    )

    result = agent.invoke(
        input={"messages": [{"role": "user", "content": "What is the weather on Tatooine?"}]},
        config=config
    )

    result = agent.invoke(
        input={"messages": [{"role": "user", "content": "Remember what is my name? Sorry for asking again."}]},
        config=config
    )

    for message in result["messages"]:
        message.pretty_print()

def test_rag_add_documents():
    loader = WebBaseLoader("https://en.wikipedia.org/wiki/World_War_I")
    docs = loader.load()
    print("docs", len(docs), len(docs[0].page_content))

    text_splitter = RecursiveCharacterTextSplitter() # chunk_size=1000, chunk_overlap=200, add_start_index=True
    all_splits = text_splitter.split_documents(docs)
    print("splits", len(all_splits), all_splits[5])

    embeddings = MistralAIEmbeddings(model="mistral-embed")
    vector_store = Chroma(
        collection_name="wikipedia_ww1",
        embedding_function=embeddings,
        persist_directory="./db/chroma_langchain_db",
    )

    ids = vector_store.add_documents(documents=all_splits)
    print(ids)

def test_rag_search():
    embeddings = MistralAIEmbeddings(model="mistral-embed")
    vector_store = Chroma(
        collection_name="wikipedia_ww1",
        embedding_function=embeddings,
        persist_directory="./db/chroma_langchain_db",
    )

    results = vector_store.similarity_search(
        "america", k=3
    )
    for result in results:
        print(result)

def main():
    # test_model()
    # test_basic_agent()
    # test_agent_memory()
    # test_rag_add_documents()
    test_rag_search()


if __name__ == "__main__":
    main()
