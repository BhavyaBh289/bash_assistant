from transformers import AutoModelForCausalLM, AutoTokenizer

# Specify the model name
model_name = "codellama/CodeLlama-7b-hf"

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Save the model and tokenizer locally
local_dir = "./CodeLlama-7b-hf-local"
tokenizer.save_pretrained(local_dir)
model.save_pretrained(local_dir)

print(f"Model and tokenizer saved to {local_dir}")
