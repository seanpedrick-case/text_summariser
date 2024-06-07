import gradio as gr
from datetime import datetime
import pandas as pd
from transformers import pipeline, AutoTokenizer
import os
from typing import Type
import gradio as gr

from llama_cpp import Llama
from huggingface_hub import hf_hub_download

PandasDataFrame = Type[pd.DataFrame]

import chatfuncs.chatfuncs as chatf
import chatfuncs.summarise_funcs as sumf

from chatfuncs.helper_functions import dummy_function, put_columns_in_df
from chatfuncs.summarise_funcs import summarise_text

# Disable cuda devices if necessary
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 

from torch import cuda, backends

# Check for torch cuda
print("Is CUDA enabled? ", cuda.is_available())
print("Is a CUDA device available on this computer?", backends.cudnn.enabled)
if cuda.is_available():
    torch_device = "cuda"
    os.system("nvidia-smi")

else: 
    torch_device =  "cpu"

print("Device used is: ", torch_device)

def create_hf_model(model_name):

    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length = chatf.context_length)

    summariser = pipeline("summarization", model=model_name, tokenizer=tokenizer) # philschmid/bart-large-cnn-samsum    

    return summariser, tokenizer, model_name

def load_model(model_type, gpu_layers, gpu_config=None, cpu_config=None, torch_device=None):
    print("Loading model ", model_type)

    # Default values inside the function
    if gpu_config is None:
        gpu_config = chatf.gpu_config
    if cpu_config is None:
        cpu_config = chatf.cpu_config
    if torch_device is None:
        torch_device = chatf.torch_device

    if model_type == "Phi 3 128k (larger, slow)":
        if torch_device == "cuda":
            gpu_config.update_gpu(gpu_layers)
            print("Loading with", gpu_config.n_gpu_layers, "model layers sent to GPU.")
        else:
            gpu_config.update_gpu(gpu_layers)
            cpu_config.update_gpu(gpu_layers)

            print("Loading with", cpu_config.n_gpu_layers, "model layers sent to GPU.")

        print(vars(gpu_config))
        print(vars(cpu_config))

        try:
            summariser = Llama(
            model_path=hf_hub_download(
            repo_id=os.environ.get("REPO_ID", "QuantFactory/Phi-3-mini-128k-instruct-GGUF"),# "QuantFactory/Phi-3-mini-128k-instruct-GGUF"), # "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF-v2"), #"microsoft/Phi-3-mini-4k-instruct-gguf"),#"TheBloke/Mistral-7B-OpenOrca-GGUF"),
            filename=os.environ.get("MODEL_FILE", "Phi-3-mini-128k-instruct.Q4_K_M.gguf") #"Phi-3-mini-128k-instruct.Q4_K_M.gguf")  #"Meta-Llama-3-8B-Instruct-v2.Q6_K.gguf") #"Phi-3-mini-4k-instruct-q4.gguf")#"mistral-7b-openorca.Q4_K_M.gguf"),
        ),
        **vars(gpu_config) # change n_gpu_layers if you have more or less VRAM 
        )
        
        except Exception as e:
            print("GPU load failed")
            print(e)
            summariser = Llama(
            model_path=hf_hub_download(
            repo_id=os.environ.get("REPO_ID", "QuantFactory/Phi-3-mini-128k-instruct-GGUF"), #"QuantFactory/Phi-3-mini-128k-instruct-GGUF"), #, "microsoft/Phi-3-mini-4k-instruct-gguf"),#"QuantFactory/Meta-Llama-3-8B-Instruct-GGUF-v2"), #"microsoft/Phi-3-mini-4k-instruct-gguf"),#"TheBloke/Mistral-7B-OpenOrca-GGUF"),
            filename=os.environ.get("MODEL_FILE", "Phi-3-mini-128k-instruct.Q4_K_M.gguf"), # "Phi-3-mini-128k-instruct.Q4_K_M.gguf") # , #"Meta-Llama-3-8B-Instruct-v2.Q6_K.gguf") #"Phi-3-mini-4k-instruct-q4.gguf"),#"mistral-7b-openorca.Q4_K_M.gguf"),
        ),
        **vars(cpu_config)
        )

        tokenizer = []

    if model_type == "Flan T5 Large Stacked Samsum 1k":
        # Huggingface chat model
        hf_checkpoint = 'stacked-summaries/flan-t5-large-stacked-samsum-1024'#'declare-lab/flan-alpaca-base' # # #

        summariser, tokenizer, model_type = create_hf_model(model_name = hf_checkpoint)

    if model_type == "Long T5 Global Base 16k Book Summary":
        # Huggingface chat model
        hf_checkpoint = 'pszemraj/long-t5-tglobal-base-16384-book-summary' #'philschmid/flan-t5-small-stacked-samsum'#'declare-lab/flan-alpaca-base' # # #
        summariser, tokenizer, model_type = create_hf_model(model_name = hf_checkpoint)

    sumf.model = summariser
    sumf.tokenizer = tokenizer
    sumf.model_type = model_type

    load_confirmation = "Finished loading model: " + model_type

    print(load_confirmation)
    return model_type, load_confirmation, model_type

# Both models are loaded on app initialisation so that users don't have to wait for the models to be downloaded
model_type = "Phi 3 128k (larger, slow)"
load_model(model_type, chatf.gpu_layers, chatf.gpu_config, chatf.cpu_config, chatf.torch_device)

model_type = "Flan T5 Large Stacked Samsum 1k"
load_model(model_type, chatf.gpu_layers, chatf.gpu_config, chatf.cpu_config, chatf.torch_device)

model_type = "Long T5 Global Base 16k Book Summary"
load_model(model_type, chatf.gpu_layers, chatf.gpu_config, chatf.cpu_config, chatf.torch_device)

today = datetime.now().strftime("%d%m%Y")
today_rev = datetime.now().strftime("%Y%m%d")


# ## Gradio app - summarise
block = gr.Blocks(theme = gr.themes.Base())

with block:  

    data_state = gr.State(pd.DataFrame())
    model_type_state = gr.State(model_type)
      
    gr.Markdown(
    """
    # Text summariser
    Enter open text below to get a summary. You can copy and paste text directly, or upload a file and specify the column that you want to summarise. The default small model will be able to summarise up to about 16,000 words, but the quality may not be great. The larger model around 900 words of better quality. Summarisation with Phi 3 128k works on up to around 4,000 words, and may give a higher quality summary, but will be slow, and it may not respect your desired maximum word count.
    """)    
    
    with gr.Tab("Summariser"):
        current_model = gr.Textbox(label="Current model", value=model_type, scale = 3)

        with gr.Accordion("Summarise open text from a file", open = True):
            in_text_df = gr.File(label="Input text from file", file_count='multiple')
            in_colname = gr.Dropdown(label="Write the column name for the open text to summarise")

        with gr.Accordion("Paste open text", open = False):
            in_text = gr.Textbox(label="Copy and paste your open text here", lines = 5)
    
        with gr.Row():
            summarise_btn = gr.Button("Summarise", variant="primary")
            stop = gr.Button(value="Interrupt processing", variant="secondary", scale=0)
            length_slider = gr.Slider(minimum = 30, maximum = 500, value = 100, step = 10, label = "Maximum length of summary")
        
        with gr.Row():
            output_single_text = gr.Textbox(label="Output example (first example in dataset)")
            output_file = gr.File(label="Output file")

    with gr.Tab("Advanced features"):
        with gr.Row():
            model_choice = gr.Radio(label="Choose a summariser model", value="Long T5 Global Base 16k Book Summary", choices = ["Long T5 Global Base 16k Book Summary", "Flan T5 Large Stacked Samsum 1k", "Phi 3 128k (larger, slow)"])
            change_model_button = gr.Button(value="Load model", scale=0)
        with gr.Accordion("Choose number of model layers to send to GPU (WARNING: please don't modify unless you are sure you have a GPU).", open = False):
            gpu_layer_choice = gr.Slider(label="Choose number of model layers to send to GPU.", value=0, minimum=0, maximum=100, step = 1, visible=True)
        with gr.Accordion("LLM parameters"):
            temp_slide = gr.Slider(minimum=0.1, value = 0.5, maximum=1, step=0.1, label="Choose temperature setting for response generation.")

        load_text = gr.Text(label="Load status")

     # Update dropdowns upon initial file load
    in_text_df.upload(put_columns_in_df, inputs=[in_text_df, in_colname], outputs=[in_colname, data_state])

    change_model_button.click(fn=load_model, inputs=[model_choice, gpu_layer_choice], outputs = [model_type_state, load_text, current_model])

    summarise_click = summarise_btn.click(fn=summarise_text, inputs=[in_text, data_state, length_slider, in_colname, model_type_state],
                       outputs=[output_single_text, output_file], api_name="summarise_single_text")
    # summarise_enter = summarise_btn.submit(fn=summarise_text, inputs=[in_text, data_state, length_slider, in_colname, model_type_state],
    #                    outputs=[output_single_text, output_file])
    
    #summarise_click = summarise_btn.click(chatf.llama_cpp_streaming, [chatbot, instruction_prompt_out, model_type_state, temp_slide], chatbot)
    
    # Stop processing if it's taking too long
    stop.click(fn=None, inputs=None, outputs=None, cancels=[summarise_click])

    # Dummy function to allow dropdown modification to work correctly (strange thing needed for Gradio 3.50, will be deprecated upon upgrading Gradio version)
    in_colname.change(dummy_function, in_colname, None)

block.queue().launch()

# def load_model(model_type, gpu_layers, gpu_config=None, cpu_config=None, torch_device=None):
#     print("Loading model ", model_type)

#     # Default values inside the function
#     if gpu_config is None:
#         gpu_config = chatf.gpu_config
#     if cpu_config is None:
#         cpu_config = chatf.cpu_config
#     if torch_device is None:
#         torch_device = chatf.torch_device

#     if model_type == "Phi 3 128k (larger, slow)":
#         hf_checkpoint = 'NousResearch/Nous-Capybara-7B-V1.9-GGUF'

#         if torch_device == "cuda":
#             gpu_config.update_gpu(gpu_layers)
#         else:
#             gpu_config.update_gpu(gpu_layers)
#             cpu_config.update_gpu(gpu_layers)

#         print("Loading with", cpu_config.gpu_layers, "model layers sent to GPU.")

#         print(vars(gpu_config))
#         print(vars(cpu_config))

#         try:
#             #model = ctransformers.AutoModelForCausalLM.from_pretrained('Aryanne/Orca-Mini-3B-gguf', model_type='llama', model_file='q5_0-orca-mini-3b.gguf', **vars(gpu_config)) # **asdict(CtransRunConfig_cpu())
#             #model = ctransformers.AutoModelForCausalLM.from_pretrained('Aryanne/Wizard-Orca-3B-gguf', model_type='llama', model_file='q4_1-wizard-orca-3b.gguf', **vars(gpu_config)) # **asdict(CtransRunConfig_cpu())
#             #model = ctransformers.AutoModelForCausalLM.from_pretrained('TheBloke/Mistral-7B-OpenOrca-GGUF', model_type='mistral', model_file='mistral-7b-openorca.Q4_K_M.gguf', **vars(gpu_config), hf=True) # **asdict(CtransRunConfig_cpu())
#             #model = ctransformers.AutoModelForCausalLM.from_pretrained('TheBloke/OpenHermes-2.5-Mistral-7B-16k-GGUF', model_type='mistral', model_file='openhermes-2.5-mistral-7b-16k.Q4_K_M.gguf', **vars(gpu_config), hf=True) # **asdict(CtransRunConfig_cpu())
#             model = ctransformers.AutoModelForCausalLM.from_pretrained('NousResearch/Nous-Capybara-7B-V1.9-GGUF', model_type='mistral', model_file='Capybara-7B-V1.9-Q5_K_M.gguf', **vars(gpu_config), hf=True) # **asdict(CtransRunConfig_cpu())


#             tokenizer = AutoTokenizer.from_pretrained("NousResearch/Nous-Capybara-7B-V1.9")
#             summariser = pipeline("text-generation", model=model, tokenizer=tokenizer)

#         except:
#             #model = ctransformers.AutoModelForCausalLM.from_pretrained('Aryanne/Orca-Mini-3B-gguf', model_type='llama', model_file='q5_0-orca-mini-3b.gguf', **vars(cpu_config)) #**asdict(CtransRunConfig_gpu())
#             #model = ctransformers.AutoModelForCausalLM.from_pretrained('Aryanne/Wizard-Orca-3B-gguf', model_type='llama', model_file='q4_1-wizard-orca-3b.gguf', **vars(cpu_config)) # **asdict(CtransRunConfig_cpu())
#             #model = ctransformers.AutoModelForCausalLM.from_pretrained('TheBloke/Mistral-7B-OpenOrca-GGUF', model_type='mistral', model_file='mistral-7b-openorca.Q4_K_M.gguf', **vars(cpu_config), hf=True) # **asdict(CtransRunConfig_cpu())
#             #model = ctransformers.AutoModelForCausalLM.from_pretrained('TheBloke/OpenHermes-2.5-Mistral-7B-16k-GGUF', model_type='mistral', model_file='openhermes-2.5-mistral-7b-16k.Q4_K_M.gguf', **vars(gpu_config), hf=True) # **asdict(CtransRunConfig_cpu())
#             model = ctransformers.AutoModelForCausalLM.from_pretrained('NousResearch/Nous-Capybara-7B-V1.9-GGUF', model_type='mistral', model_file='Capybara-7B-V1.9-Q5_K_M.gguf', **vars(gpu_config), hf=True) # **asdict(CtransRunConfig_cpu())
            
#             #tokenizer = ctransformers.AutoTokenizer.from_pretrained(model)

#             tokenizer = AutoTokenizer.from_pretrained("NousResearch/Nous-Capybara-7B-V1.9")
#             summariser = pipeline("text-generation", model=model, tokenizer=tokenizer) # model

#         #model = []
#         #tokenizer = []
#         #summariser = []

#     if model_type == "Flan T5 Large Stacked Samsum 1k":
#         # Huggingface chat model
#         hf_checkpoint = 'stacked-summaries/flan-t5-large-stacked-samsum-1024'#'declare-lab/flan-alpaca-base' # # #

#         summariser, tokenizer, model_type = create_hf_model(model_name = hf_checkpoint)

#     if model_type == "Long T5 Global Base 16k Book Summary":
#         # Huggingface chat model
#         hf_checkpoint = 'pszemraj/long-t5-tglobal-base-16384-book-summary' #'philschmid/flan-t5-small-stacked-samsum'#'declare-lab/flan-alpaca-base' # # #
#         summariser, tokenizer, model_type = create_hf_model(model_name = hf_checkpoint)

#     chatf.model = summariser
#     chatf.tokenizer = tokenizer
#     chatf.model_type = model_type

#     load_confirmation = "Finished loading model: " + model_type

#     print(load_confirmation)
#     return model_type, load_confirmation, model_type