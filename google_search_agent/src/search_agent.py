import os
import sys
import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(kit_dir, ".."))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

from langchain.agents import load_tools ,initialize_agent
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
        self.api_info, self.llm_info = self._get_config_info()
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
        
        return  api_info, llm_info
    
    def set_llm(self):
        if self.api_info == "sambaverse":
            llm = SambaverseEndpoint(
                    sambaverse_model_name=self.llm_info["sambaverse_model_name"],
                    sambaverse_api_key=os.getenv("SAMBAVERSE_API_KEY"),
                    model_kwargs={
                        "do_sample": False, 
                        "max_tokens_to_generate": self.llm_info["max_tokens_to_generate"],
                        "temperature": self.llm_info["temperature"],
                        "process_prompt": True,
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
            
    def set_agent(self):
        tools=load_tools(self.tools, self.llm)
        
        PREFIX = "[INST] <<SYS>> You are smart assistant that selects a function from list of functions based on user questions.\
        Run only one Action at a time. \nAnswer the following questions as best you can. You have access to the following tools:"

        FORMAT_INSTRUCTIONS = "Use the following format:\n\n\
        Question: the input question you must answer\n\
        Thought: you should always think about what to do\n\
        Action: the action to take, should be one of [Search, Calculator]\n\
        Action Input: the input to the action\n\
        Observation: the result of the action\n\
        ... (this Thought/Action/Action Input/Observation can repeat N times, but only one at a time)\n\
        Thought: I now know the final answer\n\
        Final Answer: the final answer to the original input question"

        SUFFIX = """Stop after each Action call and wait to get the intermediate results, do not make up multiple steps by yourself

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
                'prefix': PREFIX, 
                'format_instructions': FORMAT_INSTRUCTIONS,
                'suffix': SUFFIX
            }
        )
        
        return agent
    
    def __call__(self, query):
        return self.agent.invoke(query)