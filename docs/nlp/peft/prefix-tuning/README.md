## prefix-tuning简介

p-tuning的改进在于，把prompt的编码结果经过重参数层二次编码，然后和input的编码结果拼接，输入transformer模型。

prefix-tuning和p-tuning的区别在于不把重参数层二次编码的结果和input编码结果拼接，而是通过一种叫past_key_values的方式，放在transformer模型的每一层中。

past_key_values是transformer模型中历史计算的K和V值，用于加速推理。[NLP基础篇](nlp/transformer_construction/)介绍过，在transformer的解码模型中，当前token是不会受到后面token的影响的，后面的token都是[MASK]，当前token只会看到前面的信息，在这种情况下K和V实际上是可以看作不变的，因此没有必要每一次解码都重复计算一次前面的KV，直接缓存为past_key_value加载即可。

在prefix-tuning就是使用这样的形式把可学习的部分放到了模型的每一层，这部分内容又称做前缀。

## prefix-tuning微调

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

###  Prefix-Tuning

####  PEFT Step1 配置文件

这里建议`prefix_projection`设置为`True`，当设置为False时虽然参数低，但是loss下降缓慢，而且经过p-tuning的经验，有重参数层肯定效果更好。

```python
from peft import get_peft_model, PrefixTuningConfig, TaskType

# prefix_projection表示是否有embedding层，False则只有全连接层，True则有embedding层
config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, 
                             num_virtual_tokens=10,
                             prefix_projection=True
                           )
config
```

    PrefixTuningConfig(peft_type=<PeftType.PREFIX_TUNING: 'PREFIX_TUNING'>, auto_mapping=None, base_model_name_or_path=None, revision=None, task_type=<TaskType.CAUSAL_LM: 'CAUSAL_LM'>, inference_mode=False, num_virtual_tokens=10, token_dim=None, num_transformer_submodules=None, num_attention_heads=None, num_layers=None, encoder_hidden_size=None, prefix_projection=True)


####  PEFT Step2 创建模型


```python
model = get_peft_model(model, config)
```

```python
model.print_trainable_parameters()
```

    trainable params: 205,641,728 || all params: 1,508,753,408 || trainable%: 13.629909759249406
    

### 前期工作Step5 配置训练参数


```python
args = TrainingArguments(
    output_dir = './prefix tuning',
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


## 训练前后显存变化

| 条件                                     | 模型参数      | 可训练参数         | 初始显存占用 | 训练显存占用 |
|----------------------------------------|---------------|---------------|--------|--------| 
| baseline                               | 1,303,132,160 | 1,303,132,160 | 1.1Gb  | 23.4Gb |
| prefix-tuning(prefix_projection=False) | 1,303,132,160 | 983,040       | 1.1Gb  | 9.4Gb  |
| prefix-tuning(prefix_projection=True)  | 1,303,132,160 | 205,641,728   | 1.1Gb  | 12.3Gb |
