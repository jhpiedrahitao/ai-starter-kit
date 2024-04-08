import os
import sys
import yaml
import json
import requests

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(kit_dir, ".."))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

from langchain.agents import load_tools ,initialize_agent
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from langchain.prompts import PromptTemplate, load_prompt
from utils.sambanova_endpoint import SambaNovaEndpoint, SambaverseEndpoint
from langchain.globals import set_debug

#set_debug(True)

from dotenv import load_dotenv
load_dotenv(os.path.join(repo_dir,'.env'))
CONFIG_PATH = os.path.join(kit_dir,'config.yaml')

class SearchAgent():
    def __init__(self, tools, max_iterations) -> None:
        self.tools = tools
        self.max_iterations = max_iterations
        self.api_info, self.llm_info, self.tools_info = self._get_config_info()
        self.llm = self.set_llm()
        self.agent = self.set_agent()   
    
    def _get_config_info(self):
        """
        Loads json config file
        """
        # Read config file
        with open(CONFIG_PATH, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)
        api_info = config["api"]
        llm_info = config["llm"]
        tools_info = config["tools"]
        
        return  api_info, llm_info, tools_info
    
    def set_llm(self):
        if self.api_info == "sambaverse":
            llm = SambaverseEndpoint(
                    sambaverse_model_name=self.llm_info["sambaverse_model_name"],
                    sambaverse_api_key=os.getenv("SAMBAVERSE_API_KEY"),
                    model_kwargs={
                        "do_sample": False, 
                        "max_tokens_to_generate": self.llm_info["max_tokens_to_generate"],
                        "temperature": self.llm_info["temperature"],
                        "process_prompt": False,
                        "select_expert": self.llm_info["sambaverse_select_expert"]
                        #"stop_sequences": { "type":"str", "value":""},
                        # "repetition_penalty": {"type": "float", "value": "1"},
                        # "top_k": {"type": "int", "value": "50"},
                        # "top_p": {"type": "float", "value": "1"}
                    }
                )
            
        elif self.api_info == "sambastudio":
            llm = SambaNovaEndpoint(
                model_kwargs={
                    "do_sample": False,
                    "temperature": self.llm_info["temperature"],
                    "max_tokens_to_generate": self.llm_info["max_tokens_to_generate"],
                    #"stop_sequences": { "type":"str", "value":""},
                    # "repetition_penalty": {"type": "float", "value": "1"},
                    # "top_k": {"type": "int", "value": "50"},
                    # "top_p": {"type": "float", "value": "1"}
                }
            )
        
        return llm

    
    def load_agent_tools(self):
        self.tools=[key for key, value in self.tools.items() if value]
        tools=[]
        llm=self.llm
        
        if "conversational_query" in self.tools:
            class BasicLLMInput(BaseModel):
                query: str = Field(description="raw user interaction")   
            def queryLLM(query: str) -> str:
                """Process a query with an LLM"""
                prompt = load_prompt(os.path.join(kit_dir,"prompts/llama70b-Q&A.yaml"))
                query = prompt.format(question=query)
                return llm.invoke(query)
            askLLM = StructuredTool.from_function(
                func=queryLLM,
                name="conversational_query",
                description="process user input conversation, following conversation without factual checking",
                args_schema=BasicLLMInput,
                return_direct=True,
            )
            tools.append(askLLM)
            self.tools.remove("conversational_query")
            
        if "search_engine" in self.tools:
            
            if self.tools_info["search_tool"] == "serpapi":
                tools.extend(load_tools(["serpapi"]))
                
            elif self.tools_info["search_tool"] == "serper":
                class SerperInput(BaseModel):
                    query: str = Field(description="google search query")
                def querySerper(query: str) -> str:
                    """A search engine. Useful for when you need to answer questions about current events. Input should be a search query."""
                    url = "https://google.serper.dev/search"
                    payload = json.dumps({
                        "q": query
                    })
                    headers = {
                        'X-API-KEY': os.environ.get("SERPER_API_KEY"),
                        'Content-Type': 'application/json'
                    }
                    response = requests.post(url, headers=headers, data=payload)
                    prompt = load_prompt(os.path.join(kit_dir, "prompts/llama70b-SearchAnalysis.yaml"))
                    formated_prompt = prompt.format(question=query, context=json.dumps(response.json()))
                    return(llm.invoke(formated_prompt))
                serper = StructuredTool.from_function(
                    func=querySerper,
                    name="search_engine",
                    description="A search engine. Useful for when you need to answer questions about current events. Input should be a search query.",
                    args_schema=SerperInput,
                    return_direct=False,
                ) 
                tools.append(serper)
                
            elif self.tools_info["search_tool"] == "openserp":
                class OpenSerpInput(BaseModel):
                    query: str = Field(description="google search query")
                def queryOpenSerp(query: str) -> str:
                    """A search engine. Useful for when you need to answer questions about current events. Input should be a search query."""
                    url = "http://127.0.0.1:7000/google/search"
                    params = {
                        "lang": "EN",
                        "limit": 10,
                        "text": query
                    }
                    response = requests.get(url, params=params)
                    prompt = load_prompt(os.path.join(kit_dir, "prompts/llama70b-OpenSearchAnalysis.yaml"))
                    formated_prompt = prompt.format(question=query, context=json.dumps(response.json()))
                    return(llm.invoke(formated_prompt))
                openSerp = StructuredTool.from_function(
                    func=queryOpenSerp,
                    name="search_engine",
                    description="A search engine. Useful for when you need to answer questions about current events. Input should be a search query.",
                    args_schema=OpenSerpInput,
                    return_direct=False,
                ) 
                tools.append(openSerp)
            
            self.tools.remove("search_engine")    
            
        if "llm-math" in self.tools:   
            tools.extend(load_tools(["llm-math"], llm))
            self.tools.remove("llm-math")
            
        if self.tools:
            print(f"tools {self.tools} not implemented")
            
        return tools
            
    def set_agent(self):
        
        tools=self.load_agent_tools()
        
        PREFIX = """<s>[INST] You are a helpful assistant who can use tools one at a time to  get closer to the answer,. You have access to the following tools:"""

        FORMAT_INSTRUCTIONS = """Use the following format:
        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of {tool_names}
        Action Input: the input to the action
        Observation: the result of the action (wait for an observation, you must not provide an observation)
        ... (this Thought/Action/Action Input/Observation can repeat N times Only if an observation was provided)

        if there is not an answer based in observations yet reply (waiting for an observation) and finish your response
        if there is enough information to give a response reply
        Final Answer: the final answer to the original input question 
        if not write (...)
        """

        SUFFIX = """Don't try to make up observations  stop if an observation is needed
        Think step by step and be patient when answering.
        Only when you are done with all steps, provide the answer based on the intermediate steps.  
        Before presenting your final answer, make sure that you correctly waited for the observation steps. 

        Begin!

        Begin! <</SYS>>

        Question: {input} 
        Thought:{agent_scratchpad}[/INST]"""
        
        agent = initialize_agent(
            tools,
            self.llm,
            agent="zero-shot-react-description",
            verbose=True,
            max_iterations = self.max_iterations,
            agent_kwargs={
                #'prefix': PREFIX, 
                #'format_instructions': FORMAT_INSTRUCTIONS,
                #'suffix': SUFFIX
            }
        )
        
        return agent
    
    def __call__(self, query):
        return self.agent.invoke(query)