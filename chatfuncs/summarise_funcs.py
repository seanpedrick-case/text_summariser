import pandas as pd
import concurrent.futures
import gradio as gr
from chatfuncs.chatfuncs import model, CtransGenGenerationConfig, temperature
from datetime import datetime

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

        if model_type != "Phi 3 128k (larger, slow)":
            summarised_texts = []

            for single_text in progress.tqdm(texts_list, desc = "Summarising texts", unit = "texts"):

                summarised_text = model(single_text, max_length=length_slider)

                #print(summarised_text)

                summarised_text_str = summarised_text[0]['summary_text']

                summarised_texts.append(summarised_text_str)

                print(summarised_text_str)

                #pd.Series(summarised_texts).to_csv("summarised_texts_out.csv")

            #print(summarised_texts)

        if model_type == "Phi 3 128k (larger, slow)":

            gen_config = CtransGenGenerationConfig()
            gen_config.update_temp(temperature)

            print(gen_config)

            # Define a function that calls your model
            # def call_model(formatted_string):#, vars):
            #     return model(formatted_string)#, vars)
            
            def call_model(formatted_string, gen_config):
                """
                Calls your generation model with parameters from the CtransGenGenerationConfig object.

                Args:
                    formatted_string (str): The formatted input text for the model.
                    gen_config (CtransGenGenerationConfig): An object containing generation parameters.
                """
                # Extracting parameters from the gen_config object
                temperature = gen_config.temperature
                top_k = gen_config.top_k
                top_p = gen_config.top_p
                repeat_penalty = gen_config.repeat_penalty
                seed = gen_config.seed
                max_tokens = gen_config.max_tokens
                stream = gen_config.stream

                # Now you can call your model directly, passing the parameters:
                output = model(
                    formatted_string, 
                    temperature=temperature, 
                    top_k=top_k, 
                    top_p=top_p, 
                    repeat_penalty=repeat_penalty, 
                    seed=seed,
                    max_tokens=max_tokens,
                    stream=stream,
                )

                return output

            # Set your timeout duration (in seconds)
            timeout_duration = 300  # Adjust this value as needed

            length = str(length_slider)

            from chatfuncs.prompts import instruction_prompt_phi3

            summarised_texts = []

            for single_text in progress.tqdm(texts_list, desc = "Summarising texts", unit = "texts"):

                formatted_string = instruction_prompt_phi3.format(length=length, text=single_text)

                # Use ThreadPoolExecutor to enforce a timeout
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    #future = executor.submit(call_model, formatted_string)#, **vars(gen_config))
                    future = executor.submit(call_model, formatted_string, gen_config)
                    try:
                        output = future.result(timeout=timeout_duration)
                        # Process the output here
                    except concurrent.futures.TimeoutError:
                        error_text = f"Timeout (five minutes) occurred for text: {single_text}. Consider using a smaller model."
                        print(error_text)
                        return error_text, None

                print(output)

                output_str = output['choices'][0]['text']

                # Find the index of 'ASSISTANT: ' to select only text after this location
                # index = output_str.find('ASSISTANT: ')

                # # Check if 'ASSISTANT: ' is found in the string
                # if index != -1:
                #     # Add the length of 'ASSISTANT: ' to the index to start from the end of this substring
                #     start_index = index + len('ASSISTANT: ')
                    
                #     # Slice the string from this point to the end
                #     assistant_text = output_str[start_index:]
                # else:
                #     assistant_text = "ASSISTANT: not found in text"

                # print(assistant_text)

                #summarised_texts.append(assistant_text)

                summarised_texts.append(output_str)

                #print(summarised_text)
                
                #pd.Series(summarised_texts).to_csv("summarised_texts_out.csv")

        if text_df.empty:
            #if model_type != "Phi 3 128k (larger, slow)":
            summarised_text_out = summarised_texts[0]#.values()

            #if model_type == "Phi 3 128k (larger, slow)":
            #    summarised_text_out = summarised_texts[0]

        else: 
            summarised_text_out = summarised_texts #[d['summary_text'] for d in summarised_texts] #summarised_text[0].values()

        output_name = "summarise_output_" + today_rev + ".csv"
        output_df = pd.DataFrame({"Original text":in_text_df[in_colname_list_first],
                                    "Summarised text":summarised_text_out})

        summarised_text_out_str = str(output_df["Summarised text"][0])#.str.replace("dict_values([","").str.replace("])",""))

        output_df.to_csv(output_name, index = None)

        return summarised_text_out_str, output_name


# def summarise_text(text, text_df, length_slider, in_colname, model_type, progress=gr.Progress()):      
         
#         if text_df.empty:
#             in_colname="text"
#             in_colname_list_first = in_colname

#             in_text_df = pd.DataFrame({in_colname_list_first:[text]})
            
#         else: 
#             in_text_df = text_df
#             in_colname_list_first = in_colname

#         print(model_type)

#         texts_list = list(in_text_df[in_colname_list_first])

#         if model_type != "Phi 3 128k (larger, slow)":
#             summarised_texts = []

#             for single_text in progress.tqdm(texts_list, desc = "Summarising texts", unit = "texts"):
#                 summarised_text = chatf.model(single_text, max_length=length_slider)

#                 #print(summarised_text)

#                 summarised_text_str = summarised_text[0]['summary_text']

#                 summarised_texts.append(summarised_text_str)

#                 print(summarised_text_str)

#                 #pd.Series(summarised_texts).to_csv("summarised_texts_out.csv")

#             #print(summarised_texts)

#         if model_type == "Phi 3 128k (larger, slow)":


#             # Define a function that calls your model
#             def call_model(formatted_string, max_length=10000):
#                 return chatf.model(formatted_string, max_length=max_length)

#             # Set your timeout duration (in seconds)
#             timeout_duration = 300  # Adjust this value as needed

#             length = str(length_slider)

#             from chatfuncs.prompts import nous_capybara_prompt

#             summarised_texts = []

#             for single_text in progress.tqdm(texts_list, desc = "Summarising texts", unit = "texts"):

#                 formatted_string = nous_capybara_prompt.format(length=length, text=single_text)

#                 # Use ThreadPoolExecutor to enforce a timeout
#                 with concurrent.futures.ThreadPoolExecutor() as executor:
#                     future = executor.submit(call_model, formatted_string, 10000)
#                     try:
#                         output = future.result(timeout=timeout_duration)
#                         # Process the output here
#                     except concurrent.futures.TimeoutError:
#                         error_text = f"Timeout (five minutes) occurred for text: {single_text}. Consider using a smaller model."
#                         print(error_text)
#                         return error_text, None

#                 print(output)

#                 output_str = output[0]['generated_text']

#                 # Find the index of 'ASSISTANT: ' to select only text after this location
#                 index = output_str.find('ASSISTANT: ')

#                 # Check if 'ASSISTANT: ' is found in the string
#                 if index != -1:
#                     # Add the length of 'ASSISTANT: ' to the index to start from the end of this substring
#                     start_index = index + len('ASSISTANT: ')
                    
#                     # Slice the string from this point to the end
#                     assistant_text = output_str[start_index:]
#                 else:
#                     assistant_text = "ASSISTANT: not found in text"

#                 print(assistant_text)

#                 summarised_texts.append(assistant_text)

#                 #print(summarised_text)
                
#                 #pd.Series(summarised_texts).to_csv("summarised_texts_out.csv")

#         if text_df.empty:
#             #if model_type != "Phi 3 128k (larger, slow)":
#             summarised_text_out = summarised_texts[0]#.values()

#             #if model_type == "Phi 3 128k (larger, slow)":
#             #    summarised_text_out = summarised_texts[0]

#         else: 
#             summarised_text_out = summarised_texts #[d['summary_text'] for d in summarised_texts] #summarised_text[0].values()

#         output_name = "summarise_output_" + today_rev + ".csv"
#         output_df = pd.DataFrame({"Original text":in_text_df[in_colname_list_first],
#                                     "Summarised text":summarised_text_out})

#         summarised_text_out_str = str(output_df["Summarised text"][0])#.str.replace("dict_values([","").str.replace("])",""))

#         output_df.to_csv(output_name, index = None)

#         return summarised_text_out_str, output_name
