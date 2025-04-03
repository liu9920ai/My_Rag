import ollama
ollama.generate()
res=ollama.chat(model="qwen2.5-7b", messages=[{"role": "user","content": "你是谁的变量"}])
print(res.message.content)