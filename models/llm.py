from typing import Optional 
from langchain_ollama import ChatOllama 
from langchain_core .messages import SystemMessage ,HumanMessage 
from config import SYSTEM_PROMPT 

class LLM :

    def __init__ (self ,model_name :str ="qwen2.5:1.5b" ,temperature :float =0.7 ,repetition_penalty :float =1.0 ):
        self.llm =ChatOllama (
        model =model_name ,
        temperature =temperature ,
        repetition_penalty =repetition_penalty 
        )

    # TODO: query router to identify if the query is cyber sec, data science, general knowledge, etc. and route to different models or prompt templates accordingly

    def generate (self ,prompt :str )->str :


        messages =[
        SystemMessage (content =SYSTEM_PROMPT ),
        HumanMessage (content =prompt )
        ]


        response =self .llm .invoke (messages )
        return response .content 