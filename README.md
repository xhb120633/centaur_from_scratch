# Centaur: a foundation model of human cognition

- **Paper:** [Centaur: a foundation model of human cognition](https://marcelbinz.github.io/imgs/Centaur__preprint_.pdf)
- **Point of Contact:** [Marcel Binz](mailto:marcel.binz@helmholtz-munich.de)

Establishing a unified theory of cognition has been a major goal of psychology. While there have been previous attempts to instantiate such theories by building domain-general models, we currently do not have one model that captures the human mind in its entire breadth. Here we introduce Centaur, a computational model that can predict and simulate human behavior in any behavioral experiment that can be expressed in natural language. We derived Centaur by finetuning a state-of-the-art language model on a novel, large-scale data set called Psych-101. Psych-101 reaches an unprecedented scale, covering trial-by-trial data from over 60,000 participants performing 10,000,000 choices in 160 experiments. Centaur not only captures the behavior of held-out participants better than existing cognitive models, but also generalizes to new cover stories, structural task modifications, and entirely new domains. Furthermore, we find that the model’s internal representations become more aligned with human neural activity after finetuning. Taken together, Centaur is the first real candidate for a unified model of human cognition. We anticipate that it will have a disruptive impact on the cognitive sciences, challenging the existing paradigm for developing computational models of cognition. 

## Usage

Note that Centaur is trained on a data set in which human choices are encapsuled by "<<" and ">>" tokens. For optimal performance, it is recommended to adjust prompts accordingly.

You can use the model using HuggingFace Transformers library with 2 or more 80GB GPUs (NVIDIA Ampere or newer) with at least 150GB of free disk space to accomodate the download.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "marcelbinz/Llama-3.1-Centaur-70B"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

[test.py](https://github.com/marcelbinz/Llama-3.1-Centaur-70B/blob/main/test.py) shows an example of this type of usage.

Alternatively, you can also run the model using unsloth on a single 80GB GPU by loading the low-rank adapter directly. 

```python
from unsloth import FastLanguageModel

model_name = "marcelbinz/Llama-3.1-Centaur-70B-adapter"
model, tokenizer = FastLanguageModel.from_pretrained(
  model_name = args.model,
  max_seq_length = 32768,
  dtype = None,
  load_in_4bit = True,
)
```

[test_adapter.py](https://github.com/marcelbinz/Llama-3.1-Centaur-70B/blob/main/test_adapter.py) shows an example of this type of usage.
