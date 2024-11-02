# Centaur: a foundation model of human cognition

- **Paper:** [Centaur: a foundation model of human cognition](https://marcelbinz.github.io/imgs/Centaur__preprint_.pdf)
- **Point of Contact:** [Marcel Binz](mailto:marcel.binz@helmholtz-munich.de)

Establishing a unified theory of cognition has been a major goal of psychology. While there have been previous attempts to instantiate such theories by building domain-general models, we currently do not have one model that captures the human mind in its entire breadth. Here we introduce Centaur, a computational model that can predict and simulate human behavior in any behavioral experiment that can be expressed in natural language. We derived Centaur by finetuning a state-of-the-art language model on a novel, large-scale data set called Psych-101. Psych-101 reaches an unprecedented scale, covering trial-by-trial data from over 60,000 participants performing 10,000,000 choices in 160 experiments. Centaur not only captures the behavior of held-out participants better than existing cognitive models, but also generalizes to new cover stories, structural task modifications, and entirely new domains. Furthermore, we find that the model’s internal representations become more aligned with human neural activity after finetuning. Taken together, Centaur is the first real candidate for a unified model of human cognition. We anticipate that it will have a disruptive impact on the cognitive sciences, challenging the existing paradigm for developing computational models of cognition. 

## Usage

Note that Centaur is trained on a data set in which human choices are encapsuled by "<<" and ">>" tokens. For optimal performance, it is recommended to adjust prompts accordingly.

The recommended usage is by loading the low-rank adapter using unsloth:

```python
from unsloth import FastLanguageModel

model_name = "marcelbinz/Llama-3.1-Centaur-70B-adapter"
model, tokenizer = FastLanguageModel.from_pretrained(
  model_name = model_name,
  max_seq_length = 32768,
  dtype = None,
  load_in_4bit = True,
)

FastLanguageModel.for_inference(model)
```

This requires 80 GB GPU memory. [test_adapter.py](https://github.com/marcelbinz/Llama-3.1-Centaur-70B/blob/main/test_adapter.py) shows an example of this type of usage.

You can alternatively use the model with the HuggingFace Transformers library:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "marcelbinz/Llama-3.1-Centaur-70B"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

This requires at least 160 GB GPU memory (even more for longer prompts). [test.py](https://github.com/marcelbinz/Llama-3.1-Centaur-70B/blob/main/test.py) shows an example of this type of usage.


### Reference

```
@misc{binz2024centaurfoundationmodelhuman,
      title={Centaur: a foundation model of human cognition}, 
      author={Marcel Binz and Elif Akata and Matthias Bethge and Franziska Brändle and Fred Callaway and Julian Coda-Forno and Peter Dayan and Can Demircan and Maria K. Eckstein and Noémi Éltető and Thomas L. Griffiths and Susanne Haridi and Akshay K. Jagadish and Li Ji-An and Alexander Kipnis and Sreejan Kumar and Tobias Ludwig and Marvin Mathony and Marcelo Mattar and Alireza Modirshanechi and Surabhi S. Nath and Joshua C. Peterson and Milena Rmus and Evan M. Russek and Tankred Saanum and Natalia Scharfenberg and Johannes A. Schubert and Luca M. Schulze Buschoff and Nishad Singhi and Xin Sui and Mirko Thalmann and Fabian Theis and Vuong Truong and Vishaal Udandarao and Konstantinos Voudouris and Robert Wilson and Kristin Witte and Shuchen Wu and Dirk Wulff and Huadong Xiong and Eric Schulz},
      year={2024},
      eprint={2410.20268},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.20268}, 
}
```
