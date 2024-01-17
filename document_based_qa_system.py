import os
import chainlit as cl
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
import document_processor

chat_open_ai = ChatOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    model_name="gpt-3.5-turbo-1106",
    temperature=0,
    streaming=True
)


@cl.on_chat_start
async def main():
    file = await document_processor.get_file_from_user()

    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    docsearch = await cl.make_async(document_processor.get_docsearch)(file)

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=chat_open_ai,
        chain_type="stuff",
        retriever=docsearch.as_retriever(max_tokens_limit=4097),
    )

    msg.content = f"`{file.name}` processed. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    # prompt = message.content

    chain = cl.user_session.get("chain")
    langchain_callback = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    langchain_callback.answer_reached = True
    res = await chain.acall(message, callbacks=[langchain_callback])

    answer = res["answer"]
    sources = res["sources"].strip()
    source_elements = []

    # Get the documents from the user session
    docs = cl.user_session.get("docs")
    docs_metadata = [doc.metadata for doc in docs]
    all_sources = [m["source"] for m in docs_metadata]

    if sources:
        found_sources = []

        # Add the sources to the message
        for source in sources.split(","):
            source_name = source.strip().replace(".", "")
            # Get the index of the source
            try:
                index = all_sources.index(source_name)
            except ValueError:
                continue
            text = docs[index].page_content
            found_sources.append(source_name)
            # Create the text element referenced in the message
            source_elements.append(cl.Text(content=text, name=source_name))

        if found_sources:
            answer += f"\nSources: {', '.join(found_sources)}"
        else:
            answer += "\nNo sources found"

    if langchain_callback.has_streamed_final_answer:
        langchain_callback.final_stream.elements = source_elements
        await langchain_callback.final_stream.update()
    else:
        await cl.Message(content=answer, elements=source_elements).send()
