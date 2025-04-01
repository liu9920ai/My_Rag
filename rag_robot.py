import os
import faiss
from pathlib import Path
from langchain_community.llms import huggingface_pipeline  # 修改1
#from langchain_community.embeddings import HuggingFaceEmbeddings  # 修改2
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import List,Dict
import ollama
import torch
import sqlite3
import numpy as np
import huggingface_hub
from openai import OpenAI
from langchain_community.llms import OpenAIChat  # 修改4
from data_processing import DataProcessor  # 导入改进的数据处理器


class RAGSystem:
    def __init__(self,
                 # embedding_model: str = "BAAI/bge-M3",
                 llm_model: str = "deepseek-ai/deepseek-llm-7b-chat",
                 processed_dir: str = "processed_data",
                 use_openai: bool = False,
                 use_deepseek_api: bool = False,
                 deepseek_api_key: str = None,
                 use_ollama: bool = True,
                 ollama_model: str = "qwen2.5-7b"):
        
        # 初始化数据处理组件
        self.data_processor = DataProcessor(
            processed_dir=processed_dir
        )
        self._keep_db_connection_open()
        # 模型配置
        self.llm_model = llm_model
        self.use_openai = use_openai
        self.use_deepseek_api = use_deepseek_api
        self.deepseek_api_key = deepseek_api_key
        self.use_ollama = use_ollama
        self.ollama_model = ollama_model
        
        # 初始化检索系统
        self._init_retrieval_system()
        
        # 初始化语言模型
        self.llm = self._init_llm()

    def _init_retrieval_system(self):
        """初始化检索系统"""
        try:
            # 处理新数据并保存
            self.data_processor.process_files()
            self.data_processor.save_artifacts()
            
            # 重新加载索引确保一致性
            self.index = faiss.read_index(str(self.data_processor.index_path))
            # self.retriever = self.data_processor.search_similar()
        except Exception as e:
            print(f"检索系统初始化失败: {str(e)}")
            raise

    def _keep_db_connection_open(self):
        """确保数据库连接未关闭"""
        if self.data_processor.conn is None:
            self.data_processor.conn = sqlite3.connect(self.data_processor.output_dir / "segments.db")
        self.data_processor.conn.execute("SELECT 1")  # 测试连接
    
    def _init_llm(self):
        """初始化语言模型（选择其中一个）"""
        if self.use_ollama:
            return self._init_ollama()
        elif self.use_deepseek_api:
            return self._init_deepseek_api()
        # todo:使用openai的api
    
    def _init_ollama(self):
        """初始化Ollama客户端"""
        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{"role": "user", "content": "ping"}],
            )
            print(f"Ollama连接成功，使用模型: {self.ollama_model}")
            return ollama
        except Exception as e:
            raise ConnectionError(f"无法连接Ollama服务: {str(e)}")

    def _init_deepseek_api(self):
        """初始化DeepSeek API"""
        if not self.deepseek_api_key:
            raise ValueError("使用DeepSeek API需要API密钥")
        
        try:
            client = OpenAI(api_key=self.deepseek_api_key, base_url="https://api.deepseek.com")
            return OpenAIChat(
                openai_api_key=self.deepseek_api_key,
                model_name="deepseek-chat",
                openai_api_base="https://api.deepseek.com/v1",
                temperature=0.7
            )
        except Exception as e:
            raise ConnectionError(f"DeepSeek API连接失败: {str(e)}")

    
    def _preprocess_data(self):
        """使用DataProcessor处理数据"""
        print("开始数据预处理...")
        try:
            # 处理新文件并构建索引
            self.data_processor.process_files()
            self.data_processor.save_data()
            
            # 加载FAISS索引（添加安全参数）
            self.vector_store = faiss.read_index(self.data_processor.index_path)
            print(f"成功加载包含 {self.vector_store.index.ntotal} 个向量的索引")
        except Exception as e:
            print(f"数据预处理失败: {str(e)}")
            raise
    
    def generate_response(self, prompt: str) -> str:
        """生成回答的统一接口"""
        # 初始化LLM
        print("正在初始化大型语言模型...")
        
        # 初始化QA链
        qa_chain = self._init_qa_chain()
        
        try:
            # 执行检索增强生成
            result = qa_chain({"query": prompt})
            return result["result"]
        except Exception as e:
            return f"生成回答时出错: {str(e)}"

    def _init_qa_chain(self):
        """初始化QA链（带改进的提示模板）"""
        # 优化后的提示模板
        prompt_template = """基于以下上下文信息，请用中文给出专业、详细的回答。
        如果无法从上下文得到明确答案，请如实说明。
        
        上下文：
        {context}
        
        问题：{question}
        
        专业回答："""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        return RetrievalQA.from_chain_type(
            llm=self._load_llm(),
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )

    def _load_llm(self):
        """加载语言模型（改进的异常处理）"""
        if self.use_ollama:
            return self._init_ollama()
        elif self.use_deepseek and self.use_deepseek_api:
            return self._init_deepseek_api()
        elif self.use_openai:
            return self._init_openai()
        else:
            return self._init_local_model()

    

    def ensure_model_downloaded(self, model_name: str) -> str:
        """改进的模型下载方法"""
        model_dir = os.path.join(self.local_model_path, model_name.split('/')[-1])
        
        if os.path.exists(model_dir):
            return model_dir
            
        try:
            huggingface_hub.snapshot_download(
                repo_id=model_name,
                local_dir=model_dir,
                local_dir_use_symlinks=False,
                resume_download=True
            )
            return model_dir
        except Exception as e:
            raise RuntimeError(f"模型下载失败: {str(e)}")
        
    def query(self, question: str) -> str:
        """执行RAG查询"""
        try:
            # 检索相关上下文
            context = self._retrieve_context(question)
            
            # 构建提示
            prompt = self._build_prompt(question, context)
            
            # 生成回答
            return self.llm.generate(model = self.ollama_model,prompt = prompt).response
        except Exception as e:
            return f"查询失败: {str(e)}"

    def _retrieve_context(self, question: str) -> List[Dict]:
        """检索相关上下文"""
        query_embedding = self.data_processor.model.encode([question])[0]
        distances, indices = self.index.search(np.array([query_embedding], dtype=np.float32), 5)
        
        context = []
        cursor = self.data_processor.conn.cursor()
        for idx in indices[0]:
            cursor.execute('SELECT text FROM segments WHERE id = ?', (int(idx),))
            if row := cursor.fetchone():
                context.append(row[0])
        return context[:3]  # 返回前3个相关片段
    

    def _build_prompt(self, question: str, context: List[str]) -> str:
        """构建增强提示"""
        context_str = "\n".join([f"- {text}" for text in context])
        return f"""基于以下上下文信息，请用中文给出专业回答：
        
        上下文：
        {context_str}
        
        问题：{question}
        
        请逐步思考后回答："""
    def __del__(self):
        """析构时关闭数据库连接"""
        if hasattr(self.data_processor, 'conn') and self.data_processor.conn:
            self.data_processor.conn.close()
            
# 使用示例
if __name__ == "__main__":
    rag = RAGSystem()
    
    while True:
        question = input("\n请输入问题（输入q退出）: ")
        if question.lower() == 'q':
            break
        print("\n生成回答：")
        print(rag.query(question))