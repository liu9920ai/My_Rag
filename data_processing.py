'''
处理文件夹data_to_process中的文件
可处理文件的格式：
    .docx
    .json
    .xlsx
    .xls
    .pdf
    .txt
存在的操作：
    1.处理文件并保存为向量格式(通过数据库)
    2.从向量库中搜索相似向量文本并返回
    
'''
import re
import faiss
import sqlite3
import numpy as np
from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformer
from typing import List, Set


# 处理文档的依赖项（按需注释）
# import json
# import pdfplumber
# import pandas as pd
# from docx import Document
# from PyPDF2 import PdfReader
# from PyPDF2.errors import PdfReadError


class DataProcessor:
    # 允许的文件扩展名列表（统一小写）
    ALLOWED_EXTENSIONS = {'.pdf', '.json', '.txt', '.docx', '.xlsx', '.xls'}
    # 需要排除的临时文件前缀（如 ~$开头的Office临时文件）
    TEMP_FILE_PREFIXES = ('~$', '._', '.tmp')
    
    def __init__(self,
                 dir_to_process:str = "data_to_process",
                 processed_dir:str = "processed_data",
                 embedding_model:str = "BAAI/bge-M3",
                 # segments_path:str = "segments.json",
                 db_path: str = "segments.db",      # 改用数据库
                 faiss_idx_path:str = "faiss_index.bin",
                 batch_size: int = 32  # 新增批处理大小参数
                 ):
        """
        初始化
        
        参数：
            dir_to_process:待处理文件存放位置
            processed_dir:处理后向量存放位置
            embedding_model:文本向量化模型名
        """
        self.input_dir  = Path(dir_to_process)
        self.output_dir = Path(processed_dir)
        self.batch_size = batch_size
        
        # 确认输出文件夹存在，否则创建
        self.output_dir.mkdir(exist_ok=True)
            
        # 加载模型，本地没有就远程下载
        self.model = SentenceTransformer(
            embedding_model,
            trust_remote_code=True
        )
        
        # 获取embedding模型的输出维度
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        # # 生成分段文本库
        # self.segments_path = self.output_dir / segments_path
        
        # 从数据库中初始化分段后的文本库
        self.conn = sqlite3.connect(self.output_dir / db_path)
        self._init_db()
        
        # 初始化FAISS索引（使用ID映射）
        self.index_path = self.output_dir / faiss_idx_path

        self._init_faiss_index()
            
            
    def _init_db(self):
        cursor = self.conn.cursor()
        # 文本片段表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS segments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_name TEXT NOT NULL
            )
        ''')
        # 新增：已处理文件记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processed_files (
                file_path TEXT PRIMARY KEY,  -- 文件完整路径作为主键
                file_name TEXT NOT NULL,
                processed_time TIMESTAMP NOT NULL
            )
        ''')
        self.conn.commit()  # 提交事务使表结构生效
    
    def _init_faiss_index(self):
        """初始化或加载Faiss索引，并与数据库同步"""
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            cursor = self.conn.cursor()
            cursor.execute('SELECT id FROM segments')
            db_ids = {row[0] for row in cursor.fetchall()}
            index_ids = set()
            for i in range(self.index.ntotal):
                index_ids.add(self.index.id_map.at(i))
            missing_ids = db_ids - index_ids
            if missing_ids:
                # 重新生成缺失的向量并添加到索引
                cursor.execute('SELECT text FROM segments WHERE id IN ({})'.format(','.join('?'*len(missing_ids)), list(missing_ids)))
                texts = [row[0] for row in cursor.fetchall()]
                embeddings = self.model.encode(texts)
                self.index.add_with_ids(embeddings, np.array(list(missing_ids), dtype=np.int64))
                faiss.write_index(self.index, str(self.index_path))
        else:
            base_index = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIDMap(base_index)
        
    def _get_processed_files(self) -> Set[str]:
        """
        获取已处理过的文档名
        Returns:
            Set[str]: 已处理过的文档集合
        """
        cursor = self.conn.cursor()
        cursor.execute('SELECT file_path FROM processed_files')
        return {row[0] for row in cursor.fetchall()}

    def _is_temp_file(self, file_path: Path) -> bool:
        """
        判断文档是否时能处理的类型
        Args:
            file_path (Path): 待判断文档

        Returns:
            bool: 如果能处理输出True
        """
        return file_path.name.startswith(self.TEMP_FILE_PREFIXES)
        
    def text_split(self,
                       text: str
                       )->list[str]:
        """
        将文本分割为片段
        参数：
            text:待分割文本段
        """
        paragraphs = [p.strip() for p in re.split(r'\n\s\n',text) if p.strip]
        
        segments = []
        for para in paragraphs:
            # 文本太长就按句号分割
            if len(para) > 1000:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                sentences = [s.strip() for s in sentences if s.strip()]
                segments.extend(sentences)
            else:
                segments.append(para)
        return segments
    
    def parse_file(self,
                   file_path:Path
                   )->list[str]:
        """解析单个文件，提取文本段"""
        extension = file_path.suffix.lower()
        try:
            
            if extension == '.txt':
                # 自动检测编码
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                except UnicodeDecodeError:
                    with open(file_path, 'r', encoding='gbk') as f:
                        text = f.read()
                return self.text_split(text)
            # 处理pdf文档
            elif extension == '.pdf':
                return self._parse_pdf(file_path)
            # 处理json文档
            elif extension == '.json':
                import json
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                text = json.dumps(data, ensure_ascii=False) if isinstance(data, (dict, list)) else str(data)
                return self.text_split(text)
            
            # 处理word文档
            elif extension == '.docx':
                from docx import Document
                doc = Document(file_path)
                return self.text_split('\n\n'.join(p.text for p in doc.paragraphs))
                
            # 处理excel文档
            elif extension in ['.xlsx','.xls']:
                import pandas as pd
                with pd.ExcelFile(file_path) as xls:
                    segments = []
                    for sheet in xls.sheet_names:
                        df = pd.read_excel(xls, sheet_name=sheet, dtype=str)
                        # 处理表头和数据
                        header = ' | '.join(df.columns.astype(str))
                        segments.append(f"Sheet: {sheet}\nHeader: {header}")
                        for _, row in df.iterrows():
                            segments.append(' | '.join(row.astype(str)))
                    return self.text_split('\n'.join(segments))
            
            return []
        except Exception as e:
            print(f"解析文件 {file_path.name} 时出错: {str(e)}")
            return []
    
            
    def get_valid_files(self) -> list[Path]:
        """获取所有符合要求的文件列表"""
        valid_files = []

        # 遍历目录下所有文件（递归遍历可添加.rglob("*")）
        for file_path in self.input_dir.glob("*"):
            # 跳过目录
            if not file_path.is_file():
                continue
            # 跳过临时文件
            if self._is_temp_file(file_path):
                continue
            # 检查扩展名是否在允许列表中（不区分大小写）
            if file_path.suffix.lower() in self.ALLOWED_EXTENSIONS:
                valid_files.append(file_path)
        # 获取已处理文件列表
        processed_files = self._get_processed_files()
        
        # 分离已处理和未处理文件
        new_files = []
        skipped_files = []
        for file_path in valid_files:
            # 转换为绝对路径统一格式
            abs_path = str(file_path.resolve())
            if abs_path in processed_files:
                skipped_files.append(file_path.name)
            else:
                new_files.append(file_path)
            # 提示已跳过的文件
        if skipped_files:
            print(f"[提示] 跳过以下已处理文件: {', '.join(skipped_files)}")
            
        return new_files
    
    def _parse_pdf(self, file_path: Path) -> List[str]:
        """PDF解析专用方法"""
        text_parts = []
        # 尝试使用pdfplumber
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text.strip())
            return self.text_split('\n\n'.join(text_parts))
        except ImportError:
            pass  # 回退到PyPDF2
        except Exception as e:
            print(f"pdfplumber处理失败: {e}")
        
        # 回退到PyPDF2
        try:
            from PyPDF2 import PdfReader
            with open(file_path, 'rb') as f:
                reader = PdfReader(f)
                if reader.is_encrypted:
                    print(f"跳过加密PDF: {file_path.name}")
                    return []
                text_parts = [page.extract_text() or '' for page in reader.pages]
            return self.text_split('\n\n'.join(text_parts))
        except Exception as e:
            print(f"PyPDF2处理失败: {e}")
            return []
        
    def get_valid_files(self) -> List[Path]:
        """跳过已处理文件"""
        valid_files = []
        processed_files = self._get_processed_files()
        
        for file_path in self.input_dir.glob('*'):
            if not file_path.is_file():
                continue
            if self._is_temp_file(file_path):
                continue
            if file_path.suffix.lower() not in self.ALLOWED_EXTENSIONS:
                continue
            if str(file_path.resolve()) in processed_files:
                print(f"跳过已处理文件:{file_path.name}")
                continue
            valid_files.append(file_path)
        
        return valid_files
          
    def process_files(self):
        """
        处理文件并跳过已处理项
        """
        cursor = self.conn.cursor()
        new_files = self.get_valid_files()
        
        for file_idx, file_path in enumerate(new_files, 1):
            print(f"处理文件 ({file_idx}/{len(new_files)}): {file_path.name}")
            
            try:
                # 解析文件并过滤空内容
                segments = [seg.strip() for seg in self.parse_file(file_path) if seg.strip()]
                if not segments:
                    print(f"文件无有效内容: {file_path.name}")
                    continue
                    
                # 获取当前序列值（原子操作）
                cursor.execute("SELECT seq FROM sqlite_sequence WHERE name='segments'")
                seq_result = cursor.fetchone()
                start_id = seq_result[0] + 1 if seq_result else 1
                
                # 准备插入数据
                insert_data = [(seg, str(file_path.resolve()), file_path.name) for seg in segments]
                
                try:
                    # 批量插入并验证行数
                    cursor.executemany('''
                        INSERT INTO segments (text, file_path, file_name)
                        VALUES (?, ?, ?)
                    ''', insert_data)
                    # if cursor.rowcount != len(segments):
                    #     raise sqlite3.Error(f"插入行数不匹配，预期 {len(segments)}，实际 {cursor.rowcount}")
                    
                    # 生成连续ID（更可靠的方式）
                    seg_ids = list(range(start_id, start_id + len(segments)))
                    
                    # 批量生成向量
                    embeddings = self.model.encode(
                        segments,
                        batch_size=self.batch_size,
                        show_progress_bar=False,
                        convert_to_numpy=True
                    )
                    
                    # 更新索引（添加类型验证）
                    if embeddings.dtype != np.float32:
                        embeddings = embeddings.astype(np.float32)
                    self.index.add_with_ids(embeddings, np.array(seg_ids, dtype=np.int64))
                    
                    # 记录处理状态
                    cursor.execute('''
                        INSERT INTO processed_files 
                        (file_path, file_name, processed_time)
                        VALUES (?, ?, ?)
                    ''', (str(file_path.resolve()), file_path.name, datetime.now()))
                    
                    self.conn.commit()
                    
                except sqlite3.Error as e:
                    self.conn.rollback()
                    print(f"数据库操作失败: {file_path.name} - {str(e)}")
                    continue
                
            except Exception as e:
                print(f"处理失败: {file_path.name} - {str(e)}")
                self.conn.rollback()

        
    def save_artifacts(self):
        """保存处理结果"""
        try:
            if self.index.ntotal > 0:
                faiss.write_index(self.index, str(self.index_path))
                print(f"成功保存FAISS索引，包含 {self.index.ntotal} 个向量")
            # self.conn.close()
        except Exception as e:
            print(f"保存失败: {str(e)}")
            raise
            
    def search_similar(self, query: str, top_k: int = 5) -> List[dict]:
        """搜索最相似的向量并返回

        Args:
            query (str): 问题
            top_k (int, optional): 最相似的前k个向量. Defaults to 5.

        Returns:
            List[dict]: 返回从数据库中查询到的数据
        """
        # 生成查询向量
        query_embedding = self.model.encode([query], show_progress_bar=False)[0]
        
        # FAISS搜索
        distances, indices = self.index.search(np.array([query_embedding], dtype=np.float32), top_k)
        
        # 从数据库获取结果
        cursor = self.conn.cursor()
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            cursor.execute('''
                SELECT text, file_name FROM segments WHERE id = ?
            ''', (int(idx),))
            row = cursor.fetchone()
            if row:
                results.append({
                    'text': row[0],
                    'file': row[1],
                    'score': float(1 / (1 + distance))  # 转换为相似度分数
                })
        return results    
        
if __name__ == "__main__":
    processor = DataProcessor()
    
    # 处理新文件
    print("开始处理文件...")
    processor.process_files()
    
    # 示例搜索
    print("\n示例搜索:")
    results = processor.search_similar("人工智能")
    for i, res in enumerate(results, 1):
        print(f"{i}. [相似度: {res['score']:.2f}] {res['file']}")
        print(f"   {res['text'][:100]}...\n")
    
    # 保存数据
    processor.save_artifacts()