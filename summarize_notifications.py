import os
import json
from datetime import date, datetime

from LLM_Engine import LLM_Agent

#### Load the config file
with open("config/summarize_notifications_config.json", "r") as config_file: 
    config = json.loads(config_file.read())
    
data_dir = config['data_dir']
prompt_dir = config['prompt_dir']

## Load user-profile
with open(os.path.join(data_dir, "user_profile.json"), "r") as profile_file: 
    user_profile = json.loads(profile_file.read())

## Create an LLM Agent
agent = LLM_Agent(remember_history='False')
agent.reset_generation_args({"max_new_tokens": 50, 
                             "temperature": 0.3,
                             "do_sample": True})
    
## Get Today's date
today = date.today().strftime("%Y-%m-%d")
    
## Get Today's notifications as a single string
date_matching_notifications = [x['message'] for x in user_profile['previous_notifications'] if x['date']==today]

## Safety check to prevent hallucinations
notifications_exist_today = True
if len(date_matching_notifications)==0:
    notifications_exist_today = False

if notifications_exist_today:
    todays_notifications= '\n'.join(date_matching_notifications)

    #### Load behaviour and template prompts
    with open(os.path.join(prompt_dir, "behaviour_instructions.txt"),'r') as behaviour_file:
        behaviour_instructions = behaviour_file.read()
    
    with open(os.path.join(prompt_dir, "user_identification_template.txt"),'r') as user_id_file:
        user_identification_template = user_id_file.read()

    with open(os.path.join(prompt_dir, "notifications_template.txt"),'r') as notifications_file:
        notifications_template = notifications_file.read()

    ## Fill-in prompt templates
    user_identification_data = user_identification_template.replace('<user_name>', user_profile['name'])
   
    notifications_data = notifications_template.replace('<todays_notifications>', todays_notifications)
    
    ## Use the agent to create a summary of Today's notifications
    summary = agent.generate(notifications_data + '\n' + 
                             behaviour_instructions + '\n' +
                             user_identification_data)

else:
    summary = 'No notifications yet today!' 
    
#### Save the new post to outputs with a timestamp
if not os.path.exists(config['out_dir']):
    os.makedirs(config['out_dir']) 
    
timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

with open(os.path.join(config['out_dir'], 'notifications_summary_'+timestamp+'_.txt'), 'w') as out_file:
    out_file.write(summary)
