# AI Bootcamp in India
## Homework Assignment

### Overview

During the webinar “Building a RAG Application: From Cool Demo to Production-Ready Solution,” you learned that developing RAG applications effectively requires the use of evaluation frameworks. These frameworks help you identify issues and measure the impact of optimization techniques that you apply to address those issues.

For your homework assignment, you will build a simple question-answering system using the RAG technique and evaluate its performance using RAGAS, a RAG evaluation framework.

Once you have completed the assignment, please share the link to your repository containing the evaluation results and the code committed to the repository.

Please carefully follow the instructions below for setup and guidance on how to properly complete the assignment and report your results.


### 1. Pre-requisites
In order to complete the homework assignment you will need to have
- Python 3.10+
- OpenAI API Key


### 2. Setup
- Fork this repository from GitHub
- Install dependencies by running
```sh
pip install -r requirements.txt
```
- Rename .env-example to .env
- Set your `OPENAI_API_KEY` in the .env file
```sh
OPENAI_API_KEY=sk-...
```

### 3. Implementing a Question Answering RAG System

A question-answering system is one of the most common types of LLM (Large Language Model) applications implemented using the RAG (Retrieval-Augmented Generation) technique.

In this assignment, you will implement a RAG pipeline that answers questions based on the transcript of Andrey Karpathy's famous talk, "[[1hr Talk] Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g)."

The transcript is included in this repository at [docs/intro-to-llms-karpathy.txt](docs/intro-to-llms-karpathy.txt).

To implement a Question Answering system using a Naive RAG approach, you will need to perform the following steps:

- **Data Ingestion**:
  - Split the document into chunks.
  - Embed the chunks.
  - Store the embeddings in a vector database.
- **RAG**:
  - Embed the question.
  - Retrieve the relevant context from the vector database.
  - Prompt the LLM with the question and retrieved context to generate an answer.

While the above might sound complex, it can actually be implemented with just five lines of LangChain code:

```python
openAIEmbeddings = OpenAIEmbeddings()
loader = TextLoader("./docs/intro-to-llms-karpathy.txt")
index = VectorstoreIndexCreator(embedding=openAIEmbeddings).from_loaders([loader])
llm = ChatOpenAI(temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=index.vectorstore.as_retriever(),
    return_source_documents=True,
)

question = "What is retrieval augmented generation and how does it enhance the capabilities of large language models?"
result = qa_chain({"query": question})
```

You can use any programming language or framework to implement the RAG pipeline.

There are numerous resources available online to help you implement a RAG system using your preferred stack. 

Below is a short list of resources for some of the most popular languages and frameworks.


- Python​
    - [RAG Tutorial with LangChain​](https://python.langchain.com/v0.2/docs/tutorials/rag/)
    - [Q&A RAG with LLamaIndex​](https://docs.llamaindex.ai/en/stable/understanding/putting_it_all_together/q_and_a/#semantic-search)
- Typescript​
    - [Retrieval Q&A with Typescript LangChain​](https://js.langchain.com/v0.1/docs/modules/chains/popular/vector_db_qa/)
- Java​
    - [RAG application with Redis and Spring AI​](https://www.baeldung.com/spring-ai-redis-rag-app)
- .NET​
    - [Retrieval Augmented Generation with .NET](https://devblogs.microsoft.com/dotnet/demystifying-retrieval-augmented-generation-with-dotnet/)

### 4. Generating Answers to the Question List

To ensure fair grading of all participants' submissions, we have prepared a list of 50 questions that your RAG pipeline must answer.

You can find these 50 questions in the repository at [questions.json](questions.json).

Please write a short script that retrieves contexts from your vector database for each of the questions and generates answers. The questions, retrieved contexts, and answers should be saved in a JSON format, as shown in the [output-example.json](./results/output-example.json).

This is necessary so that we can use the outputs of your system to automatically evaluate it using RAGAS metrics.

**The JSON file containing your outputs must be pushed to this repository along with your code so that we can review it.**

**Please note that the questions cannot be altered, as this ensures that the results of all participants can be fairly compared.**

Here's an example of a simple python script that generates the answers to the provided questions and saves the results in the required format:

```python
import json

json_results = []
for question in test_questions:
  response = qa_chain.invoke({"query" : question})

  json_results.append( {
    "question" : question,
    "answer" : response["result"],
    "contexts" : [context.page_content for context in response["source_documents"]]
} )

with open('./my_rag_output.json', mode='w', encoding='utf-8') as f:
  f.write( json.dumps(json_results, indent=4) )
```

### 5. Evaluation

This repository is fully set up to evaluate the outputs of a Question Answering RAG system based on the transcript of "[Andrej Karpathy's - [1hr Talk] Intro to Large Language Models](docs/intro-to-llms-karpathy.txt)."

So, when your pipeline is ready and you have generated the answers to our question list and saved them in the required format, you can simply use the following command to run the evaluation:

```sh
python eval.py path-to-your-output.json
```

The evaluation is based on the RAGAS framework and will output scores for each of the metrics. 
The overall `ragas_score` is simply the mean value of all the other metrics.


```json
{
    "results": {
        "faithfulness": 0.9149, 
        "answer_relevancy": 0.8295, 
        "context_recall": 0.8867, 
        "context_precision": 0.8478, 
        "answer_correctness": 0.7237
    }, "ragas_score": 0.8404951600375368
}
```

After running the evaluation, the results are automatically saved in the `./results` directory:

```
results/
    - ragas-eval-run-details.csv
    - ragas-eval-scores.json
```

You can use RAG optimization techniques to improve your RAGAS metrics score.

When you're satisfied with the results, please ensure that you commit and push the following to the repository:

- The code implementing your RAG pipeline
- The JSON file containing the outputs for our question list
- The files in the `results/` directory containing the evaluation results and details.

---
### Notice on Dataset Attribution

[The transcript](docs/intro-to-llms-karpathy.txt) and [evaluation dataset](eval/eval_dataset.csv) in this project are based on content from the YouTube video [[1hr Talk] Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g) by Andrej Karpathy, licensed under Creative Commons Attribution (CC-BY). Changes have been made to format the transcript. 





