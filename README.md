## DocQuery

DocQuery is a document based Q&A chatbot. You can upload PDF or TXT file and then ask questions related to the content
of the file

### Tech Stack

- Langchain
- Chainlit
- Chroma DB

### Langchain

- Framework for simplifying the creation of applications that use LLMs.

### Chainlit

- Framework which allows the easy creation of chatbot interfaces

### Chroma DB

- Open source database
- General-purpose database (can be used both for local experiments and production deployments)
- Can be used in both in-memory mode and persistent mode
- Embedding functions can be changed

### How to run the application?

Go to the top level folder and run the following commands:

- Make sure you have set your OPENAI_API_KEY as environment variable

- Install required packages:

```
pip3 install -r requirements.txt
```

- Run the application

```
chainlit run document_based_qa_system.py -w
```
