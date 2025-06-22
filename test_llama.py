from llama_cpp import Llama

llm = Llama(
    model_path="D:/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=6  # adjust for your CPU
)

response = llm("Q: What is 2 + 2?\nA:", max_tokens=20, stop=["\n"])
print(response["choices"][0]["text"].strip())
