import json

import torch 

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 


with open("LLM_Engine/config/agent_config.json", "r") as config_file: 
    agent_config = json.loads(config_file.read())

class LLM_Agent():
    """
    An inference-only instructional LLM-Agent that will take a system-instruction and user-instruction,
    and generate a text-output, and optionally remember the history of a conversation.
    """
    def __init__(self, remember_history=False):
        self.model_ckpt = agent_config['model_ckpt']
        
        self.generation_args = agent_config['default_generation_args']

        self.remember_history = remember_history
        if remember_history:
            self.messages = []
        
        self.model = AutoModelForCausalLM.from_pretrained(self.model_ckpt,
                                                          attn_implementation="eager",
                                                          torch_dtype="auto",
                                                          trust_remote_code=True)


        self.tokenizer = AutoTokenizer.from_pretrained(self.model_ckpt)

        self.pipe = pipeline("text-generation", 
                             model=self.model, 
                             tokenizer=self.tokenizer,
                             device=agent_config['device']) 
                
        
    def reset_generation_args(self, generation_args):
        self.generation_args.update(generation_args)

    def clear_history(self):
        self.messages = []

    def generate(self, system_instruct, user_instruct = '', generation_args=None):
        if generation_args!=None:
            generation_args = self.generation_args.copy().update(generation_args)
        else:
            generation_args = self.generation_args

        if self.remember_history:
            messages = self.messages
        else:
            messages = []

        messages += [{"role": "system", "content": system_instruct}, 
                     {"role": "user", "content": user_instruct}] 

        output = self.pipe(messages, **generation_args)[0]['generated_text'] 

        if self.remember_history:
            messages += [{"role": "assistant", "content": output}]
            self.messages = messages

        return output
