from llama_cpp import Llama

model_path = r'C:\Users\Sakshi Shewale\OneDrive\Desktop\gguf\llama-2-7b.Q4_K_M.gguf'
llm = Llama(model_path=model_path)

prompt = "What is the capital of France?"
output = llm(prompt)

print(output)