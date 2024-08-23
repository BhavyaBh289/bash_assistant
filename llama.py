from llama_cpp import Llama
import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AdamW,AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

# model_path = r'C:\Users\Sakshi Shewale\OneDrive\Desktop\gguf\llama-2-7b.Q4_K_M.gguf'
model_name="/home/bh289/Documents/clg/sem 7/bash_assistant/starcoderbase-local"


model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# printing model info
# hidden_size = model.config.hidden_size
# print(hidden_size)
layer_names = model.state_dict().keys()

# for name in layer_names:
    # print(name)


# prompt = "What is the capital of France?"
# output = llm(prompt)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# print(output)
with open('data_test.json', 'r') as file:
    data = json.load(file)
# Extract
input_texts = [d['nl_command'] for d in data]
output_texts = [d['bash_code'] for d in data]

# Tokenize the inputs and outputs
inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
outputs = tokenizer(output_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # Since you're working with a language model
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=[
        "attn.c_attn",
        "attn.c_proj",
        "mlp.c_fc",
        "mlp.c_proj"
    ]
)

model = get_peft_model(model, lora_config)

optimizer = AdamW(model.parameters(), lr=5e-5)

model.train()
for epoch in range(3):
    optimizer.zero_grad()
    outputs = model(input_ids=inputs['input_ids'], labels=outputs['input_ids'])
    loss = outputs.loss
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

model.save_pretrained('fine-tuned-lora-model')
