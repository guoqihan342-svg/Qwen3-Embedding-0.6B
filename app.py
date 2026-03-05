from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# 加载模型（首次启动会自动下载约1.2GB模型文件到本地）
model_name = "Qwen/Qwen3-Embedding-0.6B"
print("正在加载模型，请等待...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float32)
model.eval()  # 推理模式，节省内存

print("模型加载完成！")

def get_embedding(text, instruction="Given a text, retrieve relevant passages that answer the query"):
    """
    instruction参数可选，用于指令感知的嵌入（提升检索效果）
    如果不需要指令，设为None或空字符串
    """
    if instruction:
        text = f"Instruct: {instruction}\nQuery: {text}"
    
    # 分词并编码
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=8192)
    
    # CPU推理，不计算梯度
    with torch.no_grad():
        outputs = model(**inputs)
        
    # 使用最后一层隐藏状态的平均池化作为嵌入向量
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    
    # 归一化（余弦相似度需要）
    embedding = embedding / np.linalg.norm(embedding)
    
    return embedding.tolist()

@app.route('/embed', methods=['POST'])
def embed():
    try:
        data = request.json
        text = data.get('text', '')
        instruction = data.get('instruction', None)
        
        if not text:
            return jsonify({'error': 'text字段不能为空'}), 400
            
        vector = get_embedding(text, instruction)
        
        return jsonify({
            'embedding': vector,
            'dimension': len(vector),
            'model': 'Qwen3-Embedding-0.6B'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model': 'Qwen3-Embedding-0.6B'})

if __name__ == '__main__':
    # 绑定0.0.0.0允许外部访问，端口8080
    app.run(host='0.0.0.0', port=8080, threaded=True)
