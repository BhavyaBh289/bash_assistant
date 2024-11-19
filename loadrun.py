import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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

# Load the fine-tuned weights
model.load_state_dict(torch.load('models/codellama-fine-tuned-nl2bash.pth', map_location="cpu"))

# Ensure the model is in evaluation mode
model.eval()

# Inference function
def run_inference(input_text):
    # Tokenize the user input
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    # Generate predictions
    output_ids = model.generate(input_ids, max_length=200, num_return_sequences=1, temperature=0.7)

    # Decode the output tokens
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text

# Example usage
if __name__ == "__main__":
    nl_command = 'List all the processes that are using more than 100MB of memory and sort them by memory usage in descending order.'
    input_text = f"[INST] Docstring: {nl_command} [/INST] Code:"
    predicted_bash_command = run_inference(input_text)
    print("Predicted Bash Command:", predicted_bash_command)
