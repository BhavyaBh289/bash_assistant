import re
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import Dataset

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
tokenizer.pad_token = tokenizer.eos_token

# Load the base model with quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="float16", bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    "codellama/CodeLlama-7b-hf", quantization_config=bnb_config, device_map="auto"
)

# Load fine-tuned weights
model.load_state_dict(torch.load('models/codellama-fine-tuned-nl2bash.pth', map_location="cpu"))
model.eval()

# Preprocess function for testing data
def preprocess_example(example):
    example_dict = {
        "srno": example.get("srno", None),
        "nl_command": example.get("nl_command", None),
        "bash_code": example.get("bash_code", None),
    }
    text = f"[INST] Docstring: {example_dict['nl_command']} [/INST] Code: {example_dict['bash_code']}"
    return {"text": text, "nl_command": example_dict["nl_command"], "bash_code": example_dict["bash_code"]}

# Function to load JSON test data
def load_json_as_dataset(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return Dataset.from_list(data)

# Load and preprocess the test dataset
test_dataset = load_json_as_dataset("test_data.json")  # Replace with your test file path
test_data = test_dataset.map(preprocess_example)

# Function to run inference
def run_inference(nl_command):
    input_text = f"[INST] Docstring: {nl_command} [/INST] Code:"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    output_ids = model.generate(input_ids, max_length=150, temperature=0.7)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text

# Testing the model
def test_model(test_data):
    correct_predictions = 0
    total_examples = len(test_data)

    for example in test_data:
        nl_command = example["nl_command"]
        actual_bash_code = example["bash_code"]

        # Run inference
        predicted_bash_code = run_inference(nl_command)

        # Print results for debugging
        print(f"NL Command: {nl_command}")
        print(f"Predicted Bash Command: {predicted_bash_code}")
        print(f"Actual Bash Command: {actual_bash_code}")
        print("-" * 80)

        # Simple string match for evaluation
        if predicted_bash_code.strip() == actual_bash_code.strip():
            correct_predictions += 1

    accuracy = correct_predictions / total_examples
    print(f"Accuracy: {accuracy:.2%}")
    return accuracy

# Run testing
if __name__ == "__main__":
    accuracy = test_model(test_data)
    print(f"Model Test Accuracy: {accuracy:.2%}")
