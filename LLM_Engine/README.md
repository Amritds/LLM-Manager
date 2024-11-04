# LLM_Engine

This package contains core LLM-logic, in support of agentic-scripts for the LLM-Manager (Parent Repository)

### LLM_Agent
A conversational/instructional agent to generate new text from a system_prompt, and optionally a user_prompt,
while optionally remembering history of the conversation. 

- You can update the generation args of the task for each use-case or use the default.

- Configure model checkpoint and default generation args in ```config/agent_config.json```

**From the parent directory directory an example usage is as follows:**
```
from LLM_Engine import LLM_Agent

agent = LLM_Agent(remember_history=True)

agent.reset_generation_args({"max_new_tokens": 100, 
                             "temperature": 0.2,
                             "do_sample": True})

agent.generate('You are an AI agent. My calender entries for today are: 1. Run a mile, 2. Meet with John, 3. File taxes',
               'What should I be doing today? The meeting with John was cancelled.')


agent.clear_history()
```

### LLM_Recommender
An LLM based recommender that will embed (candidate_sentences and profile_sentences) using an LLM, and then sort candidate_sentences according to their similarity to the average profile_sentence.

- Configure model checkpoint, max_seq_length and batch_sizes in ```config/recommender_config.json```

**From the parent directory directory an example usage is as follows:**
```
from LLM_Engine import LLM_Recommender
recommender = LLM_Recommender()

sents = ['The best way to make tea is to use boiling water', 'A dog ran across the field', 'A cat jumped from tree to tree']
profile = ['Cats are my life, I own three cats']

recommender.sort_by_recommendation(sents, profile)
```

