
from typing import TypeVar

# Model packages
import torch.cuda
from transformers import pipeline

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
repetition_penalty: float = 1.3
flan_alpaca_repetition_penalty: float = 1.3
last_n_tokens: int = 64
max_new_tokens: int = 256
seed: int = 42
reset: bool = False
stream: bool = True
threads: int = threads
batch_size:int = 256
context_length:int = 4096
sample = True


class CtransInitConfig_gpu:
    def __init__(self, temperature=temperature,
                 top_k=top_k,
                 top_p=top_p,
                 repetition_penalty=repetition_penalty,
                 last_n_tokens=last_n_tokens,
                 max_new_tokens=max_new_tokens,
                 seed=seed,
                 reset=reset,
                 stream=stream,
                 threads=threads,
                 batch_size=batch_size,
                 context_length=context_length,
                 gpu_layers=gpu_layers):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty# repetition_penalty
        self.last_n_tokens = last_n_tokens
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.reset = reset
        self.stream = stream
        self.threads = threads
        self.batch_size = batch_size
        self.context_length = context_length
        self.gpu_layers = gpu_layers
        # self.stop: list[str] = field(default_factory=lambda: [stop_string])

    def update_gpu(self, new_value):
        self.gpu_layers = new_value

class CtransInitConfig_cpu(CtransInitConfig_gpu):
    def __init__(self):
        super().__init__()
        self.gpu_layers = 0

gpu_config = CtransInitConfig_gpu()
cpu_config = CtransInitConfig_cpu()


class CtransGenGenerationConfig:
    def __init__(self, temperature=temperature,
                 top_k=top_k,
                 top_p=top_p,
                 repetition_penalty=repetition_penalty,
                 last_n_tokens=last_n_tokens,
                 seed=seed,
                 threads=threads,
                 batch_size=batch_size,
                 reset=True
                 ):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty# repetition_penalty
        self.last_n_tokens = last_n_tokens
        self.seed = seed
        self.threads = threads
        self.batch_size = batch_size
        self.reset = reset

    def update_temp(self, new_value):
        self.temperature = new_value