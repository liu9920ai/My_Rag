from openai import OpenAI
import time
client = OpenAI(api_key="", base_url="https://api.deepseek.com")

messages = [
    {"role": "system", "content": "你是一个智能助手，帮助用户解决问题。"},
]


# messages.append({"role": "user", "content": message})
while(True):
    message = input("请输入消息：")
    # 退出条件
    if message.lower() == "exit":
        break
    messages.append({"role": "user", "content": message})
    # 获取返回的内容
    response = client.chat.completions.create(
        model="",
        messages=messages,
        temperature=0.7,
        max_tokens=1000,
        n=1,
        stop=None,
    )
    
    print(response.choices[0].message.content)
    messages.append({"role": "assistant", "content": response.choices[0].message.content})
    # 继续对话