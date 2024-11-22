import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
tokenizer.pad_token = tokenizer.eos_token

# Load the base model without quantization
model = AutoModelForCausalLM.from_pretrained(
    "codellama/CodeLlama-7b-hf",
    device_map="cpu",  # Ensure the model runs on CPU
    torch_dtype=torch.float32  # Use 32-bit floats for compatibility
)

# Load the fine-tuned weights
model.load_state_dict(
    torch.load(
        '/home/bh289/Documents/clg/sem 7/bash_assistant/codellama-fine-tuned-epoc3-nl2bash.pth',
        map_location="cpu"
    )
)

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
