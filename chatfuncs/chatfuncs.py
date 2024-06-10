
from typing import TypeVar

# Model packages
import torch.cuda
from transformers import pipeline
import time

torch.cuda.empty_cache()

PandasDataFrame = TypeVar('pd.core.frame.DataFrame')

model_type = None # global variable setup

full_text = "" # Define dummy source text (full text) just to enable highlight function to load

model = [] # Define empty list for model functions to run
tokenizer = [] # Define empty list for model functions to run


# Currently set gpu_layers to 0 even with cuda due to persistent bugs in implementation with cuda
if torch.cuda.is_available():
    torch_device = "cuda"
    gpu_layers = 0
else: 
    torch_device =  "cpu"
    gpu_layers = 0

print("Running on device:", torch_device)
threads = 8 #torch.get_num_threads()
print("CPU threads:", threads)

# flan-t5-large-stacked-xsum Model parameters
temperature: float = 0.1
top_k: int = 3
top_p: float = 1
repetition_penalty: float = 1.05 #1.3
last_n_tokens: int = 64
max_new_tokens: int = 4096 # 200
seed: int = 42
reset: bool = True
stream: bool = False
threads: int = threads
batch_size:int = 256
context_length:int = 4096
sample = True


class CtransInitConfig_gpu:
    def __init__(self,
                 last_n_tokens=last_n_tokens,
                 seed=seed,
                 n_threads=threads,
                 n_batch=batch_size,
                 n_ctx=24576,
                 n_gpu_layers=gpu_layers):

        self.last_n_tokens = last_n_tokens
        self.seed = seed
        self.n_threads = n_threads
        self.n_batch = n_batch
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        # self.stop: list[str] = field(default_factory=lambda: [stop_string])

    def update_gpu(self, new_value):
        self.n_gpu_layers = new_value

class CtransInitConfig_cpu(CtransInitConfig_gpu):
    def __init__(self):
        super().__init__()
        self.n_gpu_layers = 0

gpu_config = CtransInitConfig_gpu()
cpu_config = CtransInitConfig_cpu()


class CtransGenGenerationConfig:
    def __init__(self, temperature=temperature,
                 top_k=top_k,
                 top_p=top_p,
                 repeat_penalty=repetition_penalty,
                 seed=seed,
                 stream=stream,
                 max_tokens=max_new_tokens
                 ):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repeat_penalty = repeat_penalty
        self.seed = seed
        self.max_tokens=max_tokens
        self.stream = stream

    def update_temp(self, new_value):
        self.temperature = new_value


def llama_cpp_streaming(history, full_prompt, model_type,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        sample=sample,
        repetition_penalty=repetition_penalty,
        top_p=top_p,
        top_k=top_k
):
#print("Model type is: ", model_type)

#if not full_prompt.strip():
#    if history is None:
#        history = []

#    return history

        #tokens = model.tokenize(full_prompt)

    gen_config = CtransGenGenerationConfig()
    gen_config.update_temp(temperature)

    print(vars(gen_config))

    # Pull the generated text from the streamer, and update the model output.
    start = time.time()
    NUM_TOKENS=0
    print('-'*4+'Start Generation'+'-'*4)

    output = model(
    full_prompt, **vars(gen_config))

    history[-1][1] = ""
    for out in output:

        if "choices" in out and len(out["choices"]) > 0 and "text" in out["choices"][0]:
            history[-1][1] += out["choices"][0]["text"]
            NUM_TOKENS+=1
            yield history
        else:
            print(f"Unexpected output structure: {out}") 

    time_generate = time.time() - start
    print('\n')
    print('-'*4+'End Generation'+'-'*4)
    print(f'Num of generated tokens: {NUM_TOKENS}')
    print(f'Time for complete generation: {time_generate}s')
    print(f'Tokens per secound: {NUM_TOKENS/time_generate}')
    print(f'Time per token: {(time_generate/NUM_TOKENS)*1000}ms')