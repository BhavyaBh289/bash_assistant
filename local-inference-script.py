import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

class NL2BashInference:
    def __init__(self, model_path):
        """Initialize the model and tokenizer from the saved path."""
        print("Loading model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoPeftModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"  # Automatically handle GPU/CPU placement
        )
        print("Model loaded successfully!")

    def generate_bash_command(self, nl_command, max_length=200, temperature=0.7):
        """Generate bash command from natural language input."""
        # Format the input text
        input_text = f"[INST] Docstring: {nl_command} [/INST] Code:"
        
        # Tokenize input
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        
        # Generate output
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and return the generated command
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the bash command part (everything after "Code:")
        if "Code:" in generated_text:
            generated_text = generated_text.split("Code:")[-1].strip()
            
        return generated_text

def main():
    # Path to your saved model
    model_path = "./codelamainstructfinetuned"  # Replace with your actual model path
    
    # Initialize the inference class
    nl2bash = NL2BashInference(model_path)
    
    # Interactive loop for testing
    print("\nNL2Bash Command Generator")
    print("Enter 'quit' to exit")
    print("-" * 50)
    
    while True:
        # Get input from user
        nl_command = input("\nEnter natural language command: ")
        
        if nl_command.lower() in ['quit', 'exit']:
            break
            
        try:
            # Generate bash command
            bash_command = nl2bash.generate_bash_command(nl_command)
            print("\nGenerated Bash Command:")
            print("-" * 50)
            print(bash_command)
            print("-" * 50)
        except Exception as e:
            print(f"Error generating command: {str(e)}")

if __name__ == "__main__":
    main()
