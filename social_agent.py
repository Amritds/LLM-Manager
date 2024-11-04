import os
import json
from datetime import datetime

from LLM_Engine import LLM_Agent

#### Load the config file
with open("config/social_agent_config.json", "r") as config_file: 
    config = json.loads(config_file.read())
    
data_dir = config['data_dir']
prompt_dir = config['prompt_dir']

## Load social feed
with open(os.path.join(data_dir, "social_media.json"), "r") as social_file: 
    social_feed = json.loads(social_file.read())

## Load user-profile
with open(os.path.join(data_dir, "user_profile.json"), "r") as profile_file: 
    user_profile = json.loads(profile_file.read())


#### Get information from authorized data sources
data_sources = []

if config['data_sources']['fitness']:
    last_workout_content = str(user_profile['fitness_data']['last_workout'])
    data_sources.append(('fitness', 'last workout', last_workout_content))

if config['data_sources']['purchase']:
    last_purchase_content = str(sorted(user_profile['purchases'], key=lambda x: x['date'])[-1])
    data_sources.append(('purchase', 'last purchase', last_purchase_content))


if config['data_sources']['music']:
    with open(os.path.join(data_dir, "spotify_playlists.json"), "r") as spotify_file: 
        spotify_playlists = json.loads(spotify_file.read())

    frequent_music_content = str(spotify_playlists)
    data_sources.append(('music', 'frequently listened to music', frequent_music_content))
    

#### Load behaviour and template prompts
with open(os.path.join(prompt_dir, "behaviour_instructions.txt"),'r') as behaviour_file:
    behaviour_instructions = behaviour_file.read()
    
with open(os.path.join(prompt_dir, "data_template.txt"),'r') as data_template_file:
    data_template = data_template_file.read()
    
with open(os.path.join(prompt_dir, "examples_template.txt"),'r') as examples_template_file:
    examples_template = examples_template_file.read()
    
#### Add the data items from the data sources to a single data string
data = ''
for typ, source, content in data_sources:
    data_item = data_template
    data_item = data_item.replace('<type>', typ)
    data_item = data_item.replace('<source>', source)
    data_item = data_item.replace('<content>', content)
    data = data + '\n' + data_item

    
## Define the agent 
agent = LLM_Agent(remember_history='False')

## Get example posts from social media
post_examples = social_feed[config['social_platform']]['recent_posts']
max_post_length = max([len(s.split(' ')) for s in post_examples])


## Combine the example posts from social media
example_posts = examples_template.replace('<content>', '\n'.join(post_examples))
   
## Set the generation args
agent.reset_generation_args({"max_new_tokens": max_post_length*3, 
                             "temperature": 0.5,
                             "do_sample": True})

## Generate a new post
new_post = agent.generate(behaviour_instructions + '\n' +
                          data + '\n' +
                          example_posts)


#### Save the new post to outputs with a timestamp
if not os.path.exists(config['out_dir']):
    os.makedirs(config['out_dir']) 
    
timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

with open(os.path.join(config['out_dir'], 'auto_post_'+timestamp+'_.txt'), 'w') as out_file:
    out_file.write(new_post)
    
