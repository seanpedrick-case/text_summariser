import gradio as gr
from datetime import datetime
import pandas as pd
from transformers import pipeline, AutoTokenizer
import os
from typing import Type
import gradio as gr
import ctransformers
# Concurrent futures is used to cancel processes that are taking too long
import concurrent.futures

PandasDataFrame = Type[pd.DataFrame]

import chatfuncs.chatfuncs as chatf

from chatfuncs.helper_functions import dummy_function, display_info, put_columns_in_df, put_columns_in_join_df, get_temp_folder_path, empty_folder

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

    if model_type == "Mistral Nous Capybara 4k (larger, slow)":
        hf_checkpoint = 'NousResearch/Nous-Capybara-7B-V1.9-GGUF'

        if torch_device == "cuda":
            gpu_config.update_gpu(gpu_layers)
        else:
            gpu_config.update_gpu(gpu_layers)
            cpu_config.update_gpu(gpu_layers)

        print("Loading with", cpu_config.gpu_layers, "model layers sent to GPU.")

        print(vars(gpu_config))
        print(vars(cpu_config))

        try:
            #model = ctransformers.AutoModelForCausalLM.from_pretrained('Aryanne/Orca-Mini-3B-gguf', model_type='llama', model_file='q5_0-orca-mini-3b.gguf', **vars(gpu_config)) # **asdict(CtransRunConfig_cpu())
            #model = ctransformers.AutoModelForCausalLM.from_pretrained('Aryanne/Wizard-Orca-3B-gguf', model_type='llama', model_file='q4_1-wizard-orca-3b.gguf', **vars(gpu_config)) # **asdict(CtransRunConfig_cpu())
            #model = ctransformers.AutoModelForCausalLM.from_pretrained('TheBloke/Mistral-7B-OpenOrca-GGUF', model_type='mistral', model_file='mistral-7b-openorca.Q4_K_M.gguf', **vars(gpu_config), hf=True) # **asdict(CtransRunConfig_cpu())
            #model = ctransformers.AutoModelForCausalLM.from_pretrained('TheBloke/OpenHermes-2.5-Mistral-7B-16k-GGUF', model_type='mistral', model_file='openhermes-2.5-mistral-7b-16k.Q4_K_M.gguf', **vars(gpu_config), hf=True) # **asdict(CtransRunConfig_cpu())
            model = ctransformers.AutoModelForCausalLM.from_pretrained('NousResearch/Nous-Capybara-7B-V1.9-GGUF', model_type='mistral', model_file='Capybara-7B-V1.9-Q5_K_M.gguf', **vars(gpu_config), hf=True) # **asdict(CtransRunConfig_cpu())


            tokenizer = AutoTokenizer.from_pretrained("NousResearch/Nous-Capybara-7B-V1.9")
            summariser = pipeline("text-generation", model=model, tokenizer=tokenizer)

        except:
            #model = ctransformers.AutoModelForCausalLM.from_pretrained('Aryanne/Orca-Mini-3B-gguf', model_type='llama', model_file='q5_0-orca-mini-3b.gguf', **vars(cpu_config)) #**asdict(CtransRunConfig_gpu())
            #model = ctransformers.AutoModelForCausalLM.from_pretrained('Aryanne/Wizard-Orca-3B-gguf', model_type='llama', model_file='q4_1-wizard-orca-3b.gguf', **vars(cpu_config)) # **asdict(CtransRunConfig_cpu())
            #model = ctransformers.AutoModelForCausalLM.from_pretrained('TheBloke/Mistral-7B-OpenOrca-GGUF', model_type='mistral', model_file='mistral-7b-openorca.Q4_K_M.gguf', **vars(cpu_config), hf=True) # **asdict(CtransRunConfig_cpu())
            #model = ctransformers.AutoModelForCausalLM.from_pretrained('TheBloke/OpenHermes-2.5-Mistral-7B-16k-GGUF', model_type='mistral', model_file='openhermes-2.5-mistral-7b-16k.Q4_K_M.gguf', **vars(gpu_config), hf=True) # **asdict(CtransRunConfig_cpu())
            model = ctransformers.AutoModelForCausalLM.from_pretrained('NousResearch/Nous-Capybara-7B-V1.9-GGUF', model_type='mistral', model_file='Capybara-7B-V1.9-Q5_K_M.gguf', **vars(gpu_config), hf=True) # **asdict(CtransRunConfig_cpu())
            
            #tokenizer = ctransformers.AutoTokenizer.from_pretrained(model)

            tokenizer = AutoTokenizer.from_pretrained("NousResearch/Nous-Capybara-7B-V1.9")
            summariser = pipeline("text-generation", model=model, tokenizer=tokenizer) # model

        #model = []
        #tokenizer = []
        #summariser = []

    if model_type == "Flan T5 Large Stacked Samsum 1k":
        # Huggingface chat model
        hf_checkpoint = 'stacked-summaries/flan-t5-large-stacked-samsum-1024'#'declare-lab/flan-alpaca-base' # # #

        summariser, tokenizer, model_type = create_hf_model(model_name = hf_checkpoint)

    if model_type == "Long T5 Global Base 16k Book Summary":
        # Huggingface chat model
        hf_checkpoint = 'pszemraj/long-t5-tglobal-base-16384-book-summary' #'philschmid/flan-t5-small-stacked-samsum'#'declare-lab/flan-alpaca-base' # # #
        summariser, tokenizer, model_type = create_hf_model(model_name = hf_checkpoint)

    chatf.model = summariser
    chatf.tokenizer = tokenizer
    chatf.model_type = model_type

    load_confirmation = "Finished loading model: " + model_type

    print(load_confirmation)
    return model_type, load_confirmation, model_type

# Both models are loaded on app initialisation so that users don't have to wait for the models to be downloaded
model_type = "Mistral Nous Capybara 4k (larger, slow)"
load_model(model_type, chatf.gpu_layers, chatf.gpu_config, chatf.cpu_config, chatf.torch_device)

model_type = "Flan T5 Large Stacked Samsum 1k"
load_model(model_type, chatf.gpu_layers, chatf.gpu_config, chatf.cpu_config, chatf.torch_device)

model_type = "Long T5 Global Base 16k Book Summary"
load_model(model_type, chatf.gpu_layers, chatf.gpu_config, chatf.cpu_config, chatf.torch_device)

today = datetime.now().strftime("%d%m%Y")
today_rev = datetime.now().strftime("%Y%m%d")

def summarise_text(text, text_df, length_slider, in_colname, model_type, progress=gr.Progress()):      
         
        if text_df.empty:
            in_colname="text"
            in_colname_list_first = in_colname

            in_text_df = pd.DataFrame({in_colname_list_first:[text]})
            
        else: 
            in_text_df = text_df
            in_colname_list_first = in_colname

        print(model_type)

        texts_list = list(in_text_df[in_colname_list_first])

        if model_type != "Mistral Nous Capybara 4k (larger, slow)":
            summarised_texts = []

            for single_text in progress.tqdm(texts_list, desc = "Summarising texts", unit = "texts"):
                summarised_text = chatf.model(single_text, max_length=length_slider)

                #print(summarised_text)

                summarised_text_str = summarised_text[0]['summary_text']

                summarised_texts.append(summarised_text_str)

                print(summarised_text_str)

                #pd.Series(summarised_texts).to_csv("summarised_texts_out.csv")

            #print(summarised_texts)

        if model_type == "Mistral Nous Capybara 4k (larger, slow)":


            # Define a function that calls your model
            def call_model(formatted_string, max_length=10000):
                return chatf.model(formatted_string, max_length=max_length)

            # Set your timeout duration (in seconds)
            timeout_duration = 300  # Adjust this value as needed

            length = str(length_slider)

            from chatfuncs.prompts import nous_capybara_prompt

            summarised_texts = []

            for single_text in progress.tqdm(texts_list, desc = "Summarising texts", unit = "texts"):

                formatted_string = nous_capybara_prompt.format(length=length, text=single_text)

                # Use ThreadPoolExecutor to enforce a timeout
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(call_model, formatted_string, 10000)
                    try:
                        output = future.result(timeout=timeout_duration)
                        # Process the output here
                    except concurrent.futures.TimeoutError:
                        error_text = f"Timeout (five minutes) occurred for text: {single_text}. Consider using a smaller model."
                        print(error_text)
                        return error_text, None

                print(output)

                output_str = output[0]['generated_text']

                # Find the index of 'ASSISTANT: ' to select only text after this location
                index = output_str.find('ASSISTANT: ')

                # Check if 'ASSISTANT: ' is found in the string
                if index != -1:
                    # Add the length of 'ASSISTANT: ' to the index to start from the end of this substring
                    start_index = index + len('ASSISTANT: ')
                    
                    # Slice the string from this point to the end
                    assistant_text = output_str[start_index:]
                else:
                    assistant_text = "ASSISTANT: not found in text"

                print(assistant_text)

                summarised_texts.append(assistant_text)

                #print(summarised_text)
                
                #pd.Series(summarised_texts).to_csv("summarised_texts_out.csv")

        if text_df.empty:
            #if model_type != "Mistral Nous Capybara 4k (larger, slow)":
            summarised_text_out = summarised_texts[0]#.values()

            #if model_type == "Mistral Nous Capybara 4k (larger, slow)":
            #    summarised_text_out = summarised_texts[0]

        else: 
            summarised_text_out = summarised_texts #[d['summary_text'] for d in summarised_texts] #summarised_text[0].values()

        output_name = "summarise_output_" + today_rev + ".csv"
        output_df = pd.DataFrame({"Original text":in_text_df[in_colname_list_first],
                                    "Summarised text":summarised_text_out})

        summarised_text_out_str = str(output_df["Summarised text"][0])#.str.replace("dict_values([","").str.replace("])",""))

        output_df.to_csv(output_name, index = None)

        return summarised_text_out_str, output_name

# ## Gradio app - summarise
block = gr.Blocks(theme = gr.themes.Base())

with block:  

    data_state = gr.State(pd.DataFrame())
    model_type_state = gr.State(model_type)
      
    gr.Markdown(
    """
    # Text summariser
    Enter open text below to get a summary. You can copy and paste text directly, or upload a file and specify the column that you want to summarise. The default small model will be able to summarise up to about 16,000 words, but the quality may not be great. The larger model around 900 words of better quality. Summarisation with Mistral Nous Capybara 4k works on up to around 4,000 words, and may give a higher quality summary, but will be slow, and it may not respect your desired maximum word count.
    """)    
    
    with gr.Tab("Summariser"):
        current_model = gr.Textbox(label="Current model", value=model_type, scale = 3)

        with gr.Accordion("Paste open text", open = False):
            in_text = gr.Textbox(label="Copy and paste your open text here", lines = 5)
            
        with gr.Accordion("Summarise open text from a file", open = False):
            in_text_df = gr.File(label="Input text from file", file_count='multiple')
            in_colname = gr.Dropdown(label="Write the column name for the open text to summarise")
    
        with gr.Row():
            summarise_btn = gr.Button("Summarise")
            stop = gr.Button(value="Interrupt processing", variant="secondary", scale=0)
            length_slider = gr.Slider(minimum = 30, maximum = 500, value = 100, step = 10, label = "Maximum length of summary")
        
        with gr.Row():
            output_single_text = gr.Textbox(label="Output example (first example in dataset)")
            output_file = gr.File(label="Output file")

    with gr.Tab("Advanced features"):
        #out_passages = gr.Slider(minimum=1, value = 2, maximum=10, step=1, label="Choose number of passages to retrieve from the document. Numbers greater than 2 may lead to increased hallucinations or input text being truncated.")
        #temp_slide = gr.Slider(minimum=0.1, value = 0.1, maximum=1, step=0.1, label="Choose temperature setting for response generation.")
        with gr.Row():
            model_choice = gr.Radio(label="Choose a summariser model", value="Long T5 Global Base 16k Book Summary", choices = ["Long T5 Global Base 16k Book Summary", "Flan T5 Large Stacked Samsum 1k", "Mistral Nous Capybara 4k (larger, slow)"])
            change_model_button = gr.Button(value="Load model", scale=0)
        with gr.Accordion("Choose number of model layers to send to GPU (WARNING: please don't modify unless you are sure you have a GPU).", open = False):
            gpu_layer_choice = gr.Slider(label="Choose number of model layers to send to GPU.", value=0, minimum=0, maximum=100, step = 1, visible=True)

        load_text = gr.Text(label="Load status")


     # Update dropdowns upon initial file load
    in_text_df.upload(put_columns_in_df, inputs=[in_text_df, in_colname], outputs=[in_colname, data_state])

    change_model_button.click(fn=load_model, inputs=[model_choice, gpu_layer_choice], outputs = [model_type_state, load_text, current_model])

    summarise_click = summarise_btn.click(fn=summarise_text, inputs=[in_text, data_state, length_slider, in_colname, model_type_state],
                        outputs=[output_single_text, output_file], api_name="summarise_single_text")
    summarise_enter = summarise_btn.click(fn=summarise_text, inputs=[in_text, data_state, length_slider, in_colname, model_type_state],
                        outputs=[output_single_text, output_file])
    
    # Stop processing if it's taking too long
    stop.click(fn=None, inputs=None, outputs=None, cancels=[summarise_click, summarise_enter])

    # Dummy function to allow dropdown modification to work correctly (strange thing needed for Gradio 3.50, will be deprecated upon upgrading Gradio version)
    in_colname.change(dummy_function, in_colname, None)

block.queue(concurrency_count=1).launch()
# -



