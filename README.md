
# RAG 智能问答机器人

一个基于检索增强生成（RAG）的智能问答系统，支持本地模型和API服务，可处理多种文档格式并实现精准上下文检索。

![示例截图](show_image/robot_show.png)  

---

## 功能特性

- **多格式文档支持**：支持处理 PDF、Word、Excel、TXT、JSON 等格式
- **混合模型架构**：支持本地 Ollama 模型、DeepSeek API 等多种后端
- **智能检索**：基于 FAISS 向量数据库实现高效语义检索
- **增量处理**：自动跳过已处理文件，支持增量数据更新
- **容错机制**：自动处理文档编码差异和异常文件

---

## 快速开始

### 环境要求
- Python 3.8+
- NVIDIA GPU（推荐，如需本地模型推理）

### 安装步骤
```bash
# 克隆仓库
git clone https://github.com/liu9920ai/My_Rag.git
cd My_Rag

# 安装依赖
pip install -r requirements.txt

# 安装文档处理依赖（可选）
pip install pdfplumber python-docx openpyxl
```

### 数据准备
1. 在项目根目录创建 `data_to_process` 文件夹
2. 将需要处理的文档（PDF/Word/Excel/TXT 等）放入该目录

### 运行系统
```bash
python chat_robot.py
```

---

## 配置指南

### 模型选择
在 `chat_robot.py` 中修改配置：
```python
rag = RAGSystem(
    use_deepseek_api=False,  # 切换为 True 使用 DeepSeek API
    use_ollama=True,        # 使用本地 Ollama 服务
    ollama_model="qwen2.5-7b"  # 本地模型名称
)
```

### API 密钥配置
如需使用 DeepSeek API：
1. 在[DeepSeek 平台](https://platform.deepseek.com/)获取 API Key
2. 创建 `.env` 文件：
```ini
DEEPSEEK_API_KEY=your_api_key_here
```

---

## 项目结构
```
├── data_to_process/        # 原始文档存放目录
├── processed_data/         # 处理后的向量数据库
├── chat_robot.py           # 主程序入口
├── data_processing.py      # 文档处理与向量化模块
├── rag_robot.py            # RAG 系统核心逻辑
└── ollamatest.py           # Ollama 连接测试脚本
```

---

## 使用示例
```bash

# 首次运行会自动预处理文档
>>> 用户: 人工智能有哪些主要应用领域?

>>> 助手: 人工智能主要应用领域包括...
```

---

## 高级功能

### 自定义处理规则
在 `data_processing.py` 中修改：
- `text_split()`：调整文本分割策略
- `ALLOWED_EXTENSIONS`：扩展支持的文件类型
- `batch_size`：优化向量化批处理大小

### 性能调优
```python
# 在 RAGSystem 初始化时添加
torch.set_num_threads(4)          # 设置CPU线程数
faiss.omp_set_num_threads(4)      # 设置FAISS并行度
```

---

## 常见问题

Q: 如何处理加密PDF？  
A: 系统会自动跳过加密文档并在控制台提示

Q: 如何重置处理进度？  
A: 删除 `processed_data/` 目录并重新运行

Q: Ollama 服务未响应怎么办？  
A: 请确保已正确安装并运行 [ollama](https://zhuanlan.zhihu.com/p/720546185)

欢迎加我[QQ](https://qm.qq.com/q/fUZH3NVcf6)交流学习

---

## 许可协议
本项目采用 [MIT License](LICENSE)
