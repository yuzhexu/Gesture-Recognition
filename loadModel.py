from tensorflow import keras
from keras.models import load_model

# 加载模型
model_path = 'ASL_Model/model (2).h5'  # 替换为您模型文件的实际路径
model = load_model(model_path)

# 打印模型结构，以确认模型已正确加载
model.summary()

