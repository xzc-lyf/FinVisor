from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained("/Users/xzc/Downloads/llama3.1")
model = AutoModelForCausalLM.from_pretrained("/Users/xzc/Downloads/llama3.1")

# 配置 LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, # 对于生成任务使用 Causal Language Modeling
    inference_mode=False,         # 用于训练
    r=8,                          # Bottleneck 维度
    lora_alpha=32,                # LoRA 层的 scaling factor
    lora_dropout=0.1              # Dropout 防止过拟合
)

# # 将 LoRA 应用到模型
# lora_model = get_peft_model(model, lora_config)
# print("LoRA model configured.")
#
# # 加载自定义或开源数据集
# dataset = load_dataset("path_to_your_dataset")  # 替换为数据集路径
# train_data = dataset["train"]
# val_data = dataset["validation"]
#
# # 预处理函数
# def preprocess_data(batch):
#     return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)
#
# # 应用预处理
# train_data = train_data.map(preprocess_data, batched=True)
# val_data = val_data.map(preprocess_data, batched=True)
#
# # 配置训练参数
# training_args = TrainingArguments(
#     output_dir="./lora-llama",        # 输出路径
#     per_device_train_batch_size=4,   # 根据显存设置 batch size
#     num_train_epochs=3,              # 训练轮数
#     learning_rate=1e-4,              # LoRA 的学习率通常较低
#     weight_decay=0.01,
#     logging_dir="./logs",            # 日志保存路径
#     save_strategy="epoch",           # 每个 epoch 保存模型
#     evaluation_strategy="epoch",     # 每个 epoch 评估模型
#     save_total_limit=3,              # 最多保存3个模型
#     fp16=True                        # 启用混合精度训练
# )
#
# # 创建 Trainer 实例
# trainer = Trainer(
#     model=lora_model,
#     args=training_args,
#     train_dataset=train_data,
#     eval_dataset=val_data,
#     tokenizer=tokenizer
# )
#
# # 开始训练
# trainer.train()
#
# lora_model.save_pretrained("./lora-tuned-llama")
# tokenizer.save_pretrained("./lora-tuned-llama")
# print("Model saved.")
#
# # 加载模型
# model = AutoModelForCausalLM.from_pretrained("./lora-tuned-llama")
# tokenizer = AutoTokenizer.from_pretrained("./lora-tuned-llama")
#
# # 创建生成任务的 pipeline
# generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
#
# # 测试生成
# result = generator("What are the advantages of LoRA?", max_length=100)
# print(result)