import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(kit_dir, ".."))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

import streamlit as st
from ansi2html import Ansi2HTMLConverter
from google_search_agent.src.search_agent import SearchAgent

from contextlib import contextmanager, redirect_stdout
from io import StringIO

conv = Ansi2HTMLConverter()

@contextmanager
def st_capture(output_func):
    with StringIO() as stdout, redirect_stdout(stdout):
        old_write = stdout.write
        previous_output = ''

        def new_write(string):
            nonlocal previous_output
            stdout_content = stdout.getvalue()
            if stdout_content != previous_output:
                output_func(conv.convert(stdout_content), unsafe_allow_html=True)
                previous_output = stdout_content
            return old_write(string)
        
        stdout.write = new_write
        yield


def set_agent(tools, max_iterations):
    agent = SearchAgent(tools, max_iterations)
    return agent

def handle_user_input(user_question, output_term):
    if user_question:
        with st_capture(output_term.write):
            with st.spinner("Processing..."):
                response = st.session_state.agent(user_question)
                st.session_state.chat_history.append(user_question)
                st.session_state.chat_history.append(response["output"])
    
    for ques, ans, in zip(
        st.session_state.chat_history[::2],
        st.session_state.chat_history[1::2],
    ):
        with st.chat_message("user"):
            st.write(f"{ques}")

        with st.chat_message(
            "ai",
            avatar="https://sambanova.ai/hubfs/logotype_sambanova_orange.png",
        ):
            st.write(f"{ans}")

             
def main():
    st.set_page_config(
        page_title="AI Starter Kit",
        page_icon="https://sambanova.ai/wp-content/uploads/2021/05/logo_icon-footer.svg",
    )
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "tools" not in st.session_state:
        st.session_state.tools = {
            #"llm-ask": True,
            "serpapi": True,
            "llm-math": True    
            }
    if "max_iterations" not in st.session_state:
        st.session_state.max_iterations = 4
    if "agent" not in st.session_state:
        st.session_state.agent = set_agent(
            st.session_state.tools,
            st.session_state.max_iterations
        )
    #if "sources_history" not in st.session_state:
        #st.session_state.sources_history = []
    #if "show_sources" not in st.session_state:
        #st.session_state.show_sources = True
    
    st.title(":orange[SambaNova] Google Search Agent")
    
    with st.sidebar:
        st.title("**Setup**")
        
        st.markdown("Tools Available for the agent to use")
        
        #tools["llm-ask"] = st.toggle('LLM_reasoning', True)
        st.session_state.tools["serpapi"] = st.toggle('Calculator', True)
        st.session_state.tools["llm-math"] = st.toggle('Google_Search', True)
        
        st.markdown("Max Agent iterations")
        st.session_state.max_iterations=st.number_input("set a number", value=4)     
        
        if st.button("set"):
            st.session_state.agent = set_agent(
                st.session_state.tools,
                st.session_state.max_iterations
            ) 
            
        st.title("Debugging")
        output_term = st.expander(label="see intermediate steps", expanded=False)
        output_term.write("")
        
    user_question = st.chat_input("Ask questions about data in provided sites")
    handle_user_input(user_question,output_term)

if __name__ == '__main__':
    main()