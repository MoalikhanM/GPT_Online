import os 
from apikey import apikey 

import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 

os.environ['OPENAI_API_KEY'] = apikey

llm = OpenAI(temperature=0.9) 

##################################################################################################################################
with st.sidebar: 
    st.image("https://imchatgpt.org/wp-content/uploads/2023/02/Im-Chat-GPT-Favicon-1024x600.webp")
    st.title("GPT Online Choose Topic")
    choice = st.radio("Navigation", ["Basic", "Video","Poem","Short Story", "Joke", "Image Generator"])
    st.info("By choosing specific topic GPT Online App will make a script for Title written.")

if choice == "Basic":
    # App Framework
    st.title('GPT Online')
    prompt_basic = st.text_input('Write promt here...')

    response = llm(prompt_basic)
    st.write(response)
    
    #response = sequential_chain({'topic' : prompt_basic})


if choice == "Video":
    # App Framework
    st.title('GPT Online')
    prompt = st.text_input('Write promt here...')

    # Prompt templates
    title_template = PromptTemplate(
        input_variables = ['topic'],
        template = 'write me a youtube video title about {topic}'
    )

    script_template = PromptTemplate(
        input_variables = ['title', 'wikipedia_research'],
        template = 'write me a youtube video script based on this title TITLE: {title} with this wikipedia research:{wikipedia_research}'
    )


if choice == "Poem":
    # App Framework
    st.title('GPT Online')
    prompt = st.text_input('Write promt here...')

    # Prompt templates
    title_template = PromptTemplate(
        input_variables = ['topic'],
        template = 'write me a poem title about {topic}'
    )

    script_template = PromptTemplate(
        input_variables = ['title', 'wikipedia_research'],
        template = 'write me a poem script based on this title TITLE: {title} with this wikipedia research:{wikipedia_research}'
    )

if choice == "Short Story":
    # App Framework
    st.title('GPT Online')
    prompt = st.text_input('Write promt here...')

    # Prompt templates
    title_template = PromptTemplate(
        input_variables = ['topic'],
        template = 'write me a short story title about {topic}'
    )

    script_template = PromptTemplate(
        input_variables = ['title', 'wikipedia_research'],
        template = 'write me a short story script based on this title TITLE: {title} with this wikipedia research:{wikipedia_research}'
    )

if choice == "Joke":
    # App Framework
    st.title('GPT Online')
    prompt = st.text_input('Write promt here...')

    # Prompt templates
    title_template = PromptTemplate(
        input_variables = ['topic'],
        template = 'write me a joke title about {topic}'
    )

    script_template = PromptTemplate(
        input_variables = ['title', 'wikipedia_research'],
        template = 'write me a joke script based on this title TITLE: {title} with this wikipedia research:{wikipedia_research}'
    )

if choice == "Image Generator":
    #print()
    target_file = open('../Image Generator/image_generator.py')
##################################################################################################################################

# Memory
title_memory = ConversationBufferMemory(input_key = 'topic', memory_key = 'chat_history')
script_memory = ConversationBufferMemory(input_key = 'title', memory_key = 'chat_history')

# LLMs
title_chain = LLMChain(llm = llm, prompt = title_template, verbose = True, output_key = 'title', memory = title_memory)
script_chain = LLMChain(llm = llm, prompt = script_template, verbose = True, output_key = 'script', memory = script_memory)
wiki = WikipediaAPIWrapper()

# Show stuff on the screen after prompt is entered
if prompt:
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    script = script_chain.run(title=title, wikipedia_research=wiki_research)
    
    #response = sequential_chain({'topic' : prompt})
    
    st.write(title)
    st.write(script)

    with st.expander('Title History'):
        st.info(title_memory.buffer)
    
    with st.expander('Script History'):
        st.info(script_memory.buffer)
    
    with st.expander('Wikipedia Research'):
        st.info(wiki_research)