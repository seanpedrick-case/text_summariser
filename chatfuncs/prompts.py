prompt = """<|im_start|>system
You are an AI assistant that follows instruction extremely well. Help as much as you can.
<|im_start|>user
Summarise the following text in less than {length} words.
Text: {text}
Answer:<|im_end|>"""

open_hermes_prompt = """<|im_start|>system
You are "Hermes 2", a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have.<|im_end|>
<|im_start|>user
Hello, who are you?<|im_end|>
<|im_start|>assistant
"""

test_prompt = """<|im_start|>system
You are "Hermes 2", a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have.<|im_end|>
<|im_start|>user
Summarise this text "WWJ-TV (channel 62), branded on-air as CBS Detroit, is a television station licensed to Detroit, Michigan, United States, owned and operated by the CBS television network. This is an organisation that operates and broadcasts structured contents to an audience. It is owned by the network's CBS News and Stations group alongside WKBD-TV, an independent station; the stations share studios on Eleven Mile Road in the Detroit suburb of Southfield. WWJ-TV's transmitter is located in Oak Park.
Founded as WGPR-TV in 1975 by Dr. William V. Banks and the International Free and Accepted Modern Masons as an extension of WGPR (107.5 FM), channel 62 in Detroit was the first Black-owned television station in the continental United States. Though its ambitious early programming plans catering to the Black community were not entirely successful due to economic and financial limitations, the station still produced several locally notable shows and housed a fully-staffed news department. WGPR-TV helped launch the careers of multiple local and national Black television hosts and executives, including Pat Harvey, Shaun Robinson, Sharon Dahlonega Bush, and Amyre Makupson. The original studios for WGPR-TV, still in use by the radio station, have been preserved as a museum and recognized as a historical landmark with inclusion in the National Register of Historic Places.
In 1994, when a major affiliation switch threatened to leave CBS without an affiliate in the Detroit market after multiple failures to secure a more successful station, the network bought WGPR-TV and dropped all existing programming in favor of CBS and syndicated programs, later changing the call letters to WWJ-TV. The station has made multiple unsuccessful attempts at producing local newscasts in its more than 25 years under CBS ownership. From assuming the affiliation in 1994 until 2001, from 2002 to 2009 and again from 2012 until 2023, WWJ-TV was the only station directly owned by any of the "Big Three" networks not to have any significant local news presence. A full news department, known as CBS News Detroit, began operation in January 2023 as an extension of CBS News's streaming service."<|im_end|>
<|im_start|>assistant
"""

nous_capybara_test_prompt = """USER:Summarise this text 'WWJ-TV (channel 62), branded on-air as CBS Detroit, is a television station licensed to Detroit, Michigan, United States, owned and operated by the CBS television network. This is an organisation that operates and broadcasts structured contents to an audience. It is owned by the network's CBS News and Stations group alongside WKBD-TV, an independent station; the stations share studios on Eleven Mile Road in the Detroit suburb of Southfield. WWJ-TV's transmitter is located in Oak Park.
Founded as WGPR-TV in 1975 by Dr. William V. Banks and the International Free and Accepted Modern Masons as an extension of WGPR (107.5 FM), channel 62 in Detroit was the first Black-owned television station in the continental United States. Though its ambitious early programming plans catering to the Black community were not entirely successful due to economic and financial limitations, the station still produced several locally notable shows and housed a fully-staffed news department. WGPR-TV helped launch the careers of multiple local and national Black television hosts and executives, including Pat Harvey, Shaun Robinson, Sharon Dahlonega Bush, and Amyre Makupson. The original studios for WGPR-TV, still in use by the radio station, have been preserved as a museum and recognized as a historical landmark with inclusion in the National Register of Historic Places.
In 1994, when a major affiliation switch threatened to leave CBS without an affiliate in the Detroit market after multiple failures to secure a more successful station, the network bought WGPR-TV and dropped all existing programming in favor of CBS and syndicated programs, later changing the call letters to WWJ-TV. The station has made multiple unsuccessful attempts at producing local newscasts in its more than 25 years under CBS ownership. From assuming the affiliation in 1994 until 2001, from 2002 to 2009 and again from 2012 until 2023, WWJ-TV was the only station directly owned by any of the "Big Three" networks not to have any significant local news presence. A full news department, known as CBS News Detroit, began operation in January 2023 as an extension of CBS News's streaming service.'
ASSISTANT:"""

nous_capybara_prompt = """USER:Summarise the following text in less than {length} words "{text}"
ASSISTANT:"""

instruction_prompt_phi3 = """<|user|>\n
Summarise the following text in less than {length} words: "{text}"\n
Summary:<|end|>\n
<|assistant|>"""

instruction_prompt_llama3 = """<|start_header_id|>system<|end_header_id|>\n
You are an AI assistant that follows instruction extremely well. Help as much as you can.<|eot_id|><|start_header_id|>user<|end_header_id|>\n
Summarise the following text in less than {length} words: "{text}"\n
Summary:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"""