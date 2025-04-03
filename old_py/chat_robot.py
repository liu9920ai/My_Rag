import os
from data_processing import DataProcessor
from rag_robot import RAGSystem

def main():

    # 首次运行预处理数据
    if not os.path.exists('processed_data/faiss_index.bin'):
        # 初始化数据处理器
        processor = DataProcessor()
        print("首次运行，正在预处理数据...")
        processor.process_files()
        processor.save_data()
    
    # 初始化RAG系统（使用DeepSeek API或本地模型）
    rag = RAGSystem(
        embedding_model_name="BAAI/bge-M3",
        processed_data_dir="processed_data",
        use_deepseek_api=False,  # 切换为True使用DeepSeek API
        use_ollama=True  # 使用本地Ollama服务
    )
    
    print("RAG聊天机器人已启动，输入内容开始对话（输入'exit'退出）...")
    while True:
        user_input = input("\n用户: ")
        if user_input.lower() == "exit":
            break
        
        # 直接使用RAG系统生成回答（内置检索逻辑）
        try:
            response = rag.generate_response(user_input)
            print(f"\n助手: {response}")
        except Exception as e:
            print(f"\n系统错误: {str(e)}")

if __name__ == "__main__":
    main()