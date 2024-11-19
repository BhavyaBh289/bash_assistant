import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the fine-tuned model and tokenizer
model_name = "finetune"  # Directory where your model is saved
device = "cpu"  # Force usage of CPU
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Ensure pad_token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Input command
input_text = "Find all files in the current directory tree that are not newer than some_file"

# Tokenize input
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

# Generate output
outputs = model.generate(
    inputs['input_ids'],
    attention_mask=inputs['attention_mask'],
    max_length=50,
    num_beams=5,
    no_repeat_ngram_size=2
)

# Decode output
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated Bash Command:", generated_text)
