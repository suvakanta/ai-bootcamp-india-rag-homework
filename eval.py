import os
import argparse
from dotenv import load_dotenv 
import json
import jsonschema
from jsonschema import validate
from datasets import load_dataset, Dataset, Features, Value, Sequence
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    answer_correctness,
    context_recall,
    context_precision,
)

metrics = [
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    answer_correctness,
]

# Define the JSON schema
schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "answer": {"type": "string"},
            "contexts": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["question", "answer", "contexts"],
        "additionalProperties": False
    }
}

def test_dataset():
    # Replace 'your_local_path/file.csv' with the actual file path to your CSV file
    csv_file = './eval/eval_dataset.csv'

    # Load the dataset from the local CSV file
    dataset = load_dataset('csv', data_files=csv_file)
    dataset = dataset['train'].remove_columns(['Unnamed: 0'])
    return dataset

def eval_dataset(data):
    features = Features({
        "question": Value("string"),
        "answer": Value("string"),
        "contexts": Sequence(Value("string")),
        "ground_truth": Value("string")
    })

    ds_data = {
        "question": [item["question"] for item in data],
        "answer": [item["answer"] for item in data],
        "contexts": [item["contexts"] for item in data],
        "ground_truth": [item["ground_truth"] for item in data],
    }

    dataset = Dataset.from_dict(ds_data, features=features)

    return dataset

def validate_json(json_data):
    try:
        validate(instance=json_data, schema=schema)
        print("JSON data is valid.")
        return True
    except jsonschema.exceptions.ValidationError as err:
        print("JSON data is invalid.")
        print(err)
        return False
    
def run_ragas_eval(response_dataset):
    eval_results = evaluate(response_dataset, metrics)

    results_df = eval_results.to_pandas()    
    results = { "results": eval_results, "ragas_score": sum(eval_results.values()) / len(eval_results) }

    results_df.to_csv('./results/ragas-eval-run-details.csv')

    with open('./results/ragas-eval-scores.json', mode='w', encoding='utf-8') as f:
        f.write(json.dumps(results, indent=4))

    return results


def process_file(file_path):
    # Here you can add your logic to process the file
    print(f"Processing file: {file_path}")
    # Example: reading the file and printing its content
    with open(file_path, 'r') as file:
        content = file.read()
        json_content = json.loads(content)    
        
        if(validate_json(json_content)):
            dataset = test_dataset()
            test_df = dataset.to_pandas()

            test_questions = test_df["question"].values.tolist()
            test_groundtruths = test_df["ground_truth"].values.tolist()

            for index, el in enumerate(json_content):
                if el["question"] == test_questions[index]:
                    el["ground_truth"] = test_groundtruths[index]
                else:
                    print("The results with changed questions set will not be accepted.")
                    return False
            
            print( json.dumps(json_content, indent=4) )
            response_dataset = eval_dataset(json_content)
            result = run_ragas_eval(response_dataset)

            print(f"Evaluation run results: {result}")
        else:
            print("Please provide a JSON file in the expected format.")

def main():
    load_dotenv(override=True)

    parser = argparse.ArgumentParser(description="RAGAS Evaluation tool")
    parser.add_argument('-f', '--file', type=str, required=True, help='The path to the file to be processed')

    args = parser.parse_args()
    
    # Expand the file path to handle both relative and absolute paths
    file_path = os.path.expanduser(args.file)
    file_path = os.path.abspath(file_path)

    # Check if the file exists
    if not os.path.isfile(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return

    # Process the file
    process_file(file_path)

if __name__ == '__main__':
    main()
