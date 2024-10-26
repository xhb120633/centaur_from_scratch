import json 
from lm_eval.evaluator import simple_evaluate

if __name__ == "__main__":
    results = simple_evaluate(model="hf", model_args="pretrained=Centaur-3.1/1_finetuning/centaur2-final-llama/checkpoint-2000", tasks="metabench", num_fewshot=0, batch_size=8)

    with open("Centaur-3.1/4_benchmarks/metabench/centaur-2000-results.json", "w") as outfile: 
        json.dump(results["results"], outfile)

    results = simple_evaluate(model="hf", model_args="pretrained=unsloth/Meta-Llama-3.1-70B-bnb-4bit", tasks="metabench", num_fewshot=0, batch_size=8)

    with open("Centaur-3.1/4_benchmarks/metabench/base-llama-3_1-70B-results.json", "w") as outfile: 
        json.dump(results["results"], outfile)