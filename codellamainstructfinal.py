# -*- coding: utf-8 -*-
"""codellamainstructfinal.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Gni-60tAidsiVmXdNK-BkHjJ_9Zv6znG
"""

!pip install accelerate peft bitsandbytes transformers trl datasets

from google.colab import drive
drive.mount('/content/drive')

from huggingface_hub import login

# Replace "YOUR_API_TOKEN" with your actual Hugging Face API token
login("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")
import json
from datasets import Dataset
from peft import LoraConfig, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer

def preprocess_example(example):
    example_dict = {
        "srno": example.get("srno", None),
        "nl_command": example.get("nl_command", None),
        "bash_code": example.get("bash_code", None),
    }
    text = f"[INST] Docstring: {example_dict['nl_command']} [/INST] Code: {example_dict['bash_code']}"
    return {"text": text}

def load_json_as_dataset(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return Dataset.from_list(data)

def finetune_llama_v2():
    # Load your dataset
    train_dataset = load_json_as_dataset("/content/drive/MyDrive/majorproj/data_train.json")

    # Preprocess the dataset
    train_data = train_dataset.map(preprocess_example, remove_columns=["srno", "nl_command", "bash_code"])

    tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="float16", bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        "codellama/CodeLlama-7b-Instruct-hf", quantization_config=bnb_config, device_map="auto"
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    peft_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )

    training_arguments = TrainingArguments(
        output_dir="llama3.2-finetuned-nl2bash",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        logging_steps=10,
        num_train_epochs=1,
        max_steps=300,
        fp16=True,
        push_to_hub=False,
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        peft_config=peft_config,
        dataset_text_field="text",
        args=training_arguments,
        tokenizer=tokenizer,
        packing=False,
        max_seq_length=512
    )

    trainer.train()
    return model, tokenizer

if __name__ == "__main__":
    ft_mode, og_tokenizer = finetune_llama_v2()

ft_model = ft_mode
def run_inference(input_text):

    # Tokenize the user input
    input_ids = og_tokenizer(input_text, return_tensors="pt").input_ids

    # Feed the tokenized input into the model for inference
    output_ids = ft_model.generate(input_ids, max_length=200, num_return_sequences=1, temperature=0.1)

    # Decode the output tokens to generate the predicted bash command
    output_text = og_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return output_text

if __name__ == "__main__":
    # Example usage
    nl_command = 'List all the processes that are using more than 100MB of memory and sort them by memory usage in descending order.'

    # Prepare the input for the model
    input_text = f"[INST] Docstring: {nl_command} [/INST] Code:"
    predicted_bash_command = run_inference(input_text)
    print("Predicted Bash Command:", predicted_bash_command)

test_dataset = load_json_as_dataset("/content/drive/MyDrive/majorproj/data_test.json")
test_data = test_dataset.map(preprocess_example, remove_columns=["srno", "nl_command", "bash_code"])

sample_example = test_data[16]

import re

# Input string
input_string = sample_example['text']

# Regular expression pattern to extract instruction and code
pattern = r'\[INST\] Docstring: (.+?) \[/INST\] Code: (.+)'

# Match the pattern
match = re.match(pattern, input_string)

# Extract instruction and code
if match:
    nl_command = match.group(1)
    actual_code = match.group(2)
    print("nl_command =", nl_command)
    print("code =", actual_code)
else:
    print("No match found.")

input_text = f"Bash code for {nl_command}"
inputs = og_tokenizer(input_text, return_tensors="pt").to(ft_model.device)
outputs = ft_model.generate(**inputs, max_length=150)
generated_code = og_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print the results
print("Input NL Command:", nl_command)
print("Predicted Bash Command:", generated_code)
print(f"actual_code: {actual_code}")

output_dir = "/content/drive/MyDrive/majorproj/codellama-fine-tuned-300-nl2bash"
ft_model.save_pretrained(output_dir)
og_tokenizer.save_pretrained(output_dir)

print(f"Model and tokenizer saved to {output_dir}")