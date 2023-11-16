### 自定义模型适配


```python
import torch
from torch import nn
from peft import LoraConfig, get_peft_model, PeftModel
```


```python
net1 = nn.Sequential(
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 2)
)
net1
```




    Sequential(
      (0): Linear(in_features=10, out_features=10, bias=True)
      (1): ReLU()
      (2): Linear(in_features=10, out_features=2, bias=True)
    )




```python
for name, param in net1.named_parameters():
    print(name)
```

    0.weight
    0.bias
    2.weight
    2.bias
    


```python
config = LoraConfig(target_modules=['0'])
```


```python
model = get_peft_model(net1, config)
model
```




    PeftModel(
      (base_model): LoraModel(
        (model): Sequential(
          (0): Linear(
            in_features=10, out_features=10, bias=True
            (lora_dropout): ModuleDict(
              (default): Identity()
            )
            (lora_A): ModuleDict(
              (default): Linear(in_features=10, out_features=8, bias=False)
            )
            (lora_B): ModuleDict(
              (default): Linear(in_features=8, out_features=10, bias=False)
            )
            (lora_embedding_A): ParameterDict()
            (lora_embedding_B): ParameterDict()
          )
          (1): ReLU()
          (2): Linear(in_features=10, out_features=2, bias=True)
        )
      )
    )


### 一个主模型多个LoRA


```python
net1 = nn.Sequential(
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 2)
)
net1
```

    Sequential(
      (0): Linear(in_features=10, out_features=10, bias=True)
      (1): ReLU()
      (2): Linear(in_features=10, out_features=2, bias=True)
    )


```python
config1 = LoraConfig(target_modules=["0"])
model2 = get_peft_model(net1, config1)
model2.save_pretrained("./loraA")
```

```python
config2 = LoraConfig(target_modules=["2"])
model2 = get_peft_model(net1, config2)
model2.save_pretrained("./loraB")
```

```python
net1 = nn.Sequential(
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 2)
)
net1
```

    Sequential(
      (0): Linear(in_features=10, out_features=10, bias=True)
      (1): ReLU()
      (2): Linear(in_features=10, out_features=2, bias=True)
    )

```python
model2 = PeftModel.from_pretrained(net1, model_id="./loraA/", adapter_name="loraA")
model2
```

    PeftModel(
      (base_model): LoraModel(
        (model): Sequential(
          (0): Linear(
            in_features=10, out_features=10, bias=True
            (lora_dropout): ModuleDict(
              (loraA): Identity()
            )
            (lora_A): ModuleDict(
              (loraA): Linear(in_features=10, out_features=8, bias=False)
            )
            (lora_B): ModuleDict(
              (loraA): Linear(in_features=8, out_features=10, bias=False)
            )
            (lora_embedding_A): ParameterDict()
            (lora_embedding_B): ParameterDict()
          )
          (1): ReLU()
          (2): Linear(in_features=10, out_features=2, bias=True)
        )
      )
    )




```python
model2.load_adapter("./loraB/", adapter_name="loraB")
model2
```




    PeftModel(
      (base_model): LoraModel(
        (model): Sequential(
          (0): Linear(
            in_features=10, out_features=10, bias=True
            (lora_dropout): ModuleDict(
              (loraA): Identity()
            )
            (lora_A): ModuleDict(
              (loraA): Linear(in_features=10, out_features=8, bias=False)
            )
            (lora_B): ModuleDict(
              (loraA): Linear(in_features=8, out_features=10, bias=False)
            )
            (lora_embedding_A): ParameterDict()
            (lora_embedding_B): ParameterDict()
          )
          (1): ReLU()
          (2): Linear(
            in_features=10, out_features=2, bias=True
            (lora_dropout): ModuleDict(
              (loraB): Identity()
            )
            (lora_A): ModuleDict(
              (loraB): Linear(in_features=10, out_features=8, bias=False)
            )
            (lora_B): ModuleDict(
              (loraB): Linear(in_features=8, out_features=2, bias=False)
            )
            (lora_embedding_A): ParameterDict()
            (lora_embedding_B): ParameterDict()
          )
        )
      )
    )



查看用的那个Lora


```python
model2.active_adapter
```




    'loraA'



切换lora


```python
model2.set_adapter("loraB")
```


```python
model2.active_adapter
```




    'loraB'



### 禁用适配器


```python
model2.set_adapter('loraA')
```


```python
with model2.disable_adapter():
    print(model2(torch.arange(10).view(1,10).float()))
```

    tensor([[1.4662, 0.1983]])
    
