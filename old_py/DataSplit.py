# chat_robot.py
import os
from pathlib import Path
from rag_robot import RAGSystem

def main():
    # 初始化配置参数
    config = {
        "use_ollama": True,                # 使用本地Ollama服务
        "ollama_model": "qwen2.5-7b",       # Ollama模型名称
        "use_deepseek_api": False,          # 禁用DeepSeek API
        "embedding_model_name": "BAAI/bge-M3",
        "vector_db_path": "vector_db",      # 向量数据库路径
        "processed_data_dir": "data_to_process",  # 待处理数据目录
        "local_model_path": "models"        # 本地模型缓存目录
    }

    try:
        # 初始化RAG系统（自动处理数据）
        print("正在初始化RAG系统...")
        rag = RAGSystem(**config)
        
        print("\nRAG聊天机器人已就绪，输入内容开始对话（输入'exit'退出）")
        while True:
            try:
                user_input = input("\n用户: ")
                if user_input.lower() in ("exit", "quit"):
                    break
                
                # 生成回答（自动包含上下文检索）
                response = rag.generate_response(user_input)
                print(f"\n助手: {response}")
                
            except KeyboardInterrupt:
                print("\n检测到中断指令，正在退出...")
                break
            except Exception as e:
                print(f"\n处理请求时出错: {str(e)}")

    except Exception as e:
        print(f"系统初始化失败: {str(e)}")
        # 创建必要的目录结构
        Path(config["vector_db_path"]).mkdir(exist_ok=True)
        Path(config["processed_data_dir"]).mkdir(exent_ok=True)
        print("请检查：\n1. 模型服务是否运行\n2. 数据目录是否存在\n3. API密钥配置")

if __name__ == "__main__":
    main()