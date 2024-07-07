## p-tuning简介

Prompt-tuning在使用soft promot训练模式时，loss下降缓慢，因此更多时候会选择hard prompt，但是要相信，存在，即有道理。soft prompt在p tuning上得到了改善。

p-tuning只支持soft prompt，它在prompt tuning的基础上，对promopt部分进一步编码，之前是tokenizer(prompt)，现在是embedding(tokenizer(prompt))，这部分编码后的prompt再和tokenize(inputs)进行拼接。而实现embedding的方式是通过LSTM（2层LSTM+1层MLP）或MLP（3层MLP）网络层。对tokenized的inputs再编码的神经网络层又叫重参数层。

## p-tuning微调

所以代码直接复用prompt-tuning即可，只需要改动PEFT模型创建部分。

### 前期工作Step1 导入工具包


```python
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
```

### 前期工作Step2 加载数据集


```python
dataset = Dataset.load_from_disk('alpaca_data_zh/')
```

### 前期工作Step3 数据预处理


```python
tokenizer = AutoTokenizer.from_pretrained('bloom-1b4-zh/')
```


```python
def process_func(example, MAX_LENGTH = 256):
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer("\n".join(['Human: ' + example['instruction'], example['input']]).strip() + '\n\nAssistant: ')
    response = tokenizer(example['output'] + tokenizer.eos_token)
    input_ids = instruction['input_ids'] + response['input_ids']
    attention_mask = instruction['attention_mask'] + response['attention_mask']
    labels = [-100] * len(instruction['input_ids']) + response['input_ids']
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

tokenized_dataset = dataset.map(process_func, remove_columns=dataset.column_names)
```



### 前期工作Step4 导入模型


```python
model = AutoModelForCausalLM.from_pretrained('bloom-1b4-zh/')
```

###  P-Tuning

####  PEFT Step1 配置文件


```python
from peft import PromptEncoderConfig, get_peft_model, TaskType, PromptEncoderReparameterizationType

config = PromptEncoderConfig(task_type=TaskType.CAUSAL_LM, 
                             num_virtual_tokens=10
                           )

# 这里再提供一个LSTM版本的Config
config = PromptEncoderConfig(task_type=TaskType.CAUSAL_LM, 
                             num_virtual_tokens=10,
                             encoder_reparameterization_type=PromptEncoderReparameterizationType.LSTM,
                             # 设置为LSTM后还可以控制LSTM网络的参数
                             # encoder_dropout=0.1,
                             # encoder_num_layers=2,
                             # encoder_hidden_size=1024
                           )
```


####  PEFT Step2 创建模型


```python
model = get_peft_model(model, config)
```


```python
model.print_trainable_parameters()
```

MLP可训练参数占比0.95%

    trainable params: 12,609,536 || all params: 1,315,721,216 || trainable%: 0.9583744524797569
    

LSTM可训练参数占比12.9%

    trainable params: 193,030,144 || all params: 1,496,141,824 || trainable%: 12.901861367923367

### 前期工作Step5 配置训练参数


```python
args = TrainingArguments(
    output_dir = './p tuning',
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    logging_steps=10,
    num_train_epochs=1
)
```

### 前期工作Step6 配置trainer


```python
trainer = Trainer(
    args = args,
    train_dataset=tokenized_dataset,
    model = model,
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
)
```

### 前期工作Step7 模型训练


```python
trainer.train()
```

在这里虽然用的soft prompt，但是可以明显感受到loss的下降。

## 训练前后显存变化

| 条件              | 模型参数      | 可训练参数  | 初始显存占用  | 训练显存占用 |
|-----------------|---------------|--------|---------|--------| 
| baseline        | 1,303,132,160 | 1,303,132,160 | 1.1Gb   | 23.4Gb |
| p-tuning(MLP)   | 1,303,132,160 | 12,609,536 | 1.1Gb   | 8.8Gb  |
| p-tuning(LSTM)  | 1,303,132,160 | 193,030,144 | 1.1Gb   | 14.5Gb |