import gradio as gr
from datetime import datetime
import pandas as pd
from transformers import pipeline
# # Load in packages

# +
import os

# Need to overwrite version of gradio present in Huggingface spaces as it doesn't have like buttons/avatars (Oct 2023)
#os.system("pip uninstall -y gradio")
os.system("pip install gradio==3.50.0")

from typing import TypeVar
#from langchain.embeddings import HuggingFaceEmbeddings#, HuggingFaceInstructEmbeddings
#from langchain.vectorstores import FAISS
import gradio as gr

from transformers import AutoTokenizer

# Alternative model sources
import ctransformers

PandasDataFrame = TypeVar('pd.core.frame.DataFrame')

import chatfuncs.chatfuncs as chatf

# Disable cuda devices if necessary
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 

def create_hf_model(model_name):

    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length = chatf.context_length)

    summariser = pipeline("summarization", model=model_name, tokenizer=tokenizer) # philschmid/bart-large-cnn-samsum

    #from transformers import AutoModelForSeq2SeqLM,  AutoModelForCausalLM
    
    #     if torch_device == "cuda":
    #         if "flan" in model_name:
    #             model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto")
    #         else:
    #             model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    #     else:
    #         if "flan" in model_name:
    #             model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    #         else: 
    #             model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    

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

    if model_type == "Mistral Open Orca (larger, slow)":
        hf_checkpoint = 'TheBloke/MistralLite-7B-GGUF'

        if torch_device == "cuda":
            gpu_config.update_gpu(gpu_layers)
        else:
            gpu_config.update_gpu(gpu_layers)
            cpu_config.update_gpu(gpu_layers)

        print("Loading with", cpu_config.gpu_layers, "model layers sent to GPU.")

        print(vars(gpu_config))
        print(vars(cpu_config))

        #try:
            #model = ctransformers.AutoModelForCausalLM.from_pretrained('Aryanne/Orca-Mini-3B-gguf', model_type='llama', model_file='q5_0-orca-mini-3b.gguf', **vars(gpu_config)) # **asdict(CtransRunConfig_cpu())
            #model = ctransformers.AutoModelForCausalLM.from_pretrained('Aryanne/Wizard-Orca-3B-gguf', model_type='llama', model_file='q4_1-wizard-orca-3b.gguf', **vars(gpu_config)) # **asdict(CtransRunConfig_cpu())
            #model = ctransformers.AutoModelForCausalLM.from_pretrained('TheBloke/Mistral-7B-OpenOrca-GGUF', model_type='mistral', model_file='mistral-7b-openorca.Q4_K_M.gguf', **vars(gpu_config), hf=True) # **asdict(CtransRunConfig_cpu())
            
        #except:
            #model = ctransformers.AutoModelForCausalLM.from_pretrained('Aryanne/Orca-Mini-3B-gguf', model_type='llama', model_file='q5_0-orca-mini-3b.gguf', **vars(cpu_config)) #**asdict(CtransRunConfig_gpu())
            #model = ctransformers.AutoModelForCausalLM.from_pretrained('Aryanne/Wizard-Orca-3B-gguf', model_type='llama', model_file='q4_1-wizard-orca-3b.gguf', **vars(cpu_config)) # **asdict(CtransRunConfig_cpu())
            #model = ctransformers.AutoModelForCausalLM.from_pretrained('TheBloke/Mistral-7B-OpenOrca-GGUF', model_type='mistral', model_file='mistral-7b-openorca.Q4_K_M.gguf', **vars(cpu_config), hf=True) # **asdict(CtransRunConfig_cpu())
            
        #tokenizer = ctransformers.AutoTokenizer.from_pretrained(model)
        #summariser = pipeline("text-generation", model=model, tokenizer=tokenizer)

        model = []
        tokenizer = []
        summariser = []

    if model_type == "flan-t5-large-stacked-samsum":
        # Huggingface chat model
        hf_checkpoint = 'stacked-summaries/flan-t5-large-stacked-samsum-1024'#'declare-lab/flan-alpaca-base' # # #

        summariser, tokenizer, model_type = create_hf_model(model_name = hf_checkpoint)

    if model_type == "flan-t5-small-stacked-samsum":
        # Huggingface chat model
        hf_checkpoint = 'stacked-summaries/flan-t5-small-stacked-samsum-1024' #'philschmid/flan-t5-small-stacked-samsum'#'declare-lab/flan-alpaca-base' # # #


        summariser, tokenizer, model_type = create_hf_model(model_name = hf_checkpoint)

    chatf.model = summariser
    chatf.tokenizer = tokenizer
    chatf.model_type = model_type

    load_confirmation = "Finished loading model: " + model_type

    print(load_confirmation)
    return model_type, load_confirmation, model_type

# Both models are loaded on app initialisation so that users don't have to wait for the models to be downloaded
#model_type = "Mistral Open Orca (larger, slow)"
#load_model(model_type, chatf.gpu_layers, chatf.gpu_config, chatf.cpu_config, chatf.torch_device)

model_type = "flan-t5-large-stacked-samsum"
load_model(model_type, chatf.gpu_layers, chatf.gpu_config, chatf.cpu_config, chatf.torch_device)

model_type = "flan-t5-small-stacked-samsum"
load_model(model_type, 0, chatf.gpu_config, chatf.cpu_config, chatf.torch_device)

today = datetime.now().strftime("%d%m%Y")
today_rev = datetime.now().strftime("%Y%m%d")

def summarise_text(text, text_df, length_slider, in_colnames, model_type):      
         
        if text_df == None:
            in_colnames="text"
            in_colnames_list_first = in_colnames

            in_text_df = pd.DataFrame({in_colnames_list_first:[text]})
            
        else: 
            in_text_df = pd.read_csv(text_df.name, delimiter = ",", low_memory=False, encoding='cp1252')
            in_colnames_list_first = in_colnames.tolist()[0][0]

        if model_type != "Mistral Open Orca (larger, slow)":
            summarised_text = chatf.model(list(in_text_df[in_colnames_list_first]), max_length=length_slider)

        if model_type == "Mistral Open Orca (larger, slow)":

            length = str(length_slider)

            prompt = """<|im_start|>system
You are an AI assistant that follows instruction extremely well. Help as much as you can.
<|im_start|>user
Summarise the following text in less than {length} words.
Text: {text}
Answer:<|im_end|>"""

            formatted_string = prompt.format(length=length, text=text)

            print(formatted_string)

            #summarised_text = chatf.model(formatted_string, max_new_tokens=length_slider)

            summarised_text = "Mistral Open Orca summaries currently not working. Sorry!"

        if text_df == None:
            if model_type != "Mistral Open Orca (larger, slow)":
                summarised_text_out = summarised_text[0].values()

            if model_type == "Mistral Open Orca (larger, slow)":
                summarised_text_out = summarised_text

        else: 
            summarised_text_out = [d['summary_text'] for d in summarised_text] #summarised_text[0].values()

        output_name = "summarise_output_" + today_rev + ".csv"
        output_df = pd.DataFrame({"Original text":in_text_df[in_colnames_list_first],
                                  "Summarised text":summarised_text_out})
        
        summarised_text_out_str = str(output_df["Summarised text"][0])#.str.replace("dict_values([","").str.replace("])",""))
        
        output_df.to_csv(output_name, index = None)
        
        return summarised_text_out_str, output_name

# ## Gradio app - summarise
block = gr.Blocks(theme = gr.themes.Base())

with block:  

    model_type_state = gr.State(model_type)
      
    gr.Markdown(
    """
    # Text summariser
    Enter open text below to get a summary. You can copy and paste text directly, or upload a file and specify the column that you want to summarise. Note that summarisation with Mistral Open Orca is still in development and does not currently work.
    """)    
    
    with gr.Tab("Summariser"):
        current_model = gr.Textbox(label="Current model", value=model_type, scale = 3)

        with gr.Accordion("Paste open text", open = False):
            in_text = gr.Textbox(label="Copy and paste your open text here", lines = 5)
            
        with gr.Accordion("Summarise open text from a file", open = False):
            in_text_df = gr.File(label="Input text from file")
            in_colnames = gr.Dataframe(label="Write the column name for the open text to summarise",
                                    type="numpy", row_count=(1,"fixed"), col_count = (1,"fixed"),
                                headers=["Open text column name"])#, "Address column name 2", "Address column name 3", "Address column name 4"])
    
        with gr.Row():
            summarise_btn = gr.Button("Summarise")
            length_slider = gr.Slider(minimum = 30, maximum = 200, value = 100, step = 10, label = "Maximum length of summary")
        
        with gr.Row():
            output_single_text = gr.Textbox(label="Output example (first example in dataset)")
            output_file = gr.File(label="Output file")

    with gr.Tab("Advanced features"):
        #out_passages = gr.Slider(minimum=1, value = 2, maximum=10, step=1, label="Choose number of passages to retrieve from the document. Numbers greater than 2 may lead to increased hallucinations or input text being truncated.")
        #temp_slide = gr.Slider(minimum=0.1, value = 0.1, maximum=1, step=0.1, label="Choose temperature setting for response generation.")
        with gr.Row():
            model_choice = gr.Radio(label="Choose a summariser model", value="flan-t5-small-stacked-samsum", choices = ["flan-t5-small-stacked-samsum", "flan-t5-large-stacked-samsum", "Mistral Open Orca (larger, slow)"])
            change_model_button = gr.Button(value="Load model", scale=0)
        with gr.Accordion("Choose number of model layers to send to GPU (WARNING: please don't modify unless you are sure you have a GPU).", open = False):
            gpu_layer_choice = gr.Slider(label="Choose number of model layers to send to GPU.", value=0, minimum=0, maximum=5, step = 1, visible=True)

        load_text = gr.Text(label="Load status")


    change_model_button.click(fn=load_model, inputs=[model_choice, gpu_layer_choice], outputs = [model_type_state, load_text, current_model])

    summarise_btn.click(fn=summarise_text, inputs=[in_text, in_text_df, length_slider, in_colnames, model_type_state],
                        outputs=[output_single_text, output_file], api_name="summarise_single_text")

block.queue(concurrency_count=1).launch()
# -



