import os
import re
import json

import requests

from bs4 import BeautifulSoup
from LLM_Engine import LLM_Agent
from LLM_Engine import LLM_Recommender

#### Load the config file
with open("config/playlist_curator_config.json", "r") as config_file: 
    config = json.loads(config_file.read())
    
data_dir = config['data_dir']
prompt_dir = config['prompt_dir']

read_pg= requests.get(config['top_charts_url'])    

## Get top chart songs
top_chart_songs = [x.get_text().strip() for x in BeautifulSoup(read_pg.text, "html.parser").findAll(attrs={"class":"a-font-primary-bold-s", "id":"title-of-a-story"})][3:]

## Get top chart artists
top_chart_artists = [x.get_text().strip() for x in BeautifulSoup(read_pg.text, "html.parser").findAll(attrs={"class":"c-label a-no-trucate a-font-primary-s lrv-u-font-size-14@mobile-max u-line-height-normal@mobile-max u-letter-spacing-0021 lrv-u-display-block a-truncate-ellipsis-2line u-max-width-330 u-max-width-230@tablet-only"})]


## Combine artists and songs to create candidate tracks
top_tracks = []
for artist, song in zip(top_chart_artists, top_chart_songs):
    top_tracks.append(artist+' - '+song)

## Load playlists created by the user  
with open(os.path.join(data_dir, "spotify_playlists.json"), "r") as spotify_file: 
    spotify_playlists = json.loads(spotify_file.read())
        
#### Load behaviour and template prompts
with open(os.path.join(prompt_dir, "behaviour_instructions.txt"),'r') as behaviour_file:
    behaviour_instructions = behaviour_file.read()
    
with open(os.path.join(prompt_dir, "example_template.txt"),'r') as example_template_file:
    example_template = example_template_file.read()
    

## Helper to generate a new playlist name (if generate_names set True in config)
def generate_new_playlist_name_from_example(agent, example_name):    
    example_data = example_template.replace('<playlist_name>', example_name)
    
    new_playlist_name = agent.generate(behaviour_instructions + '\n' +
                                       example_data)
    
    return new_playlist_name

## Optionally load an LLM Agent (if you have enough GPU memory)
if config['generate_names']:
    agent = LLM_Agent(remember_history=False)

    agent.reset_generation_args({"max_new_tokens": 7, 
                                     "temperature": 0.5,
                                     "do_sample": True})

## Load an LLM Recommender
recommender = LLM_Recommender()

## Generate new playlists with (optionally) new names
new_playlists = []
for playlist in spotify_playlists['playlists'][:config['max_num_new_playlists']]:
    
    # Optionally generate a name
    if config['generate_names']:
        new_name = generate_new_playlist_name_from_example(agent, playlist['name'])
    else:
        new_name = 'Try something similar to your : ' + playlist['name'] + ' playlist'
        
    # Sort top tracks by recommendation
    recs = recommender.sort_by_recommendation(top_tracks, playlist['tracks'])

    # Create the new playlist
    new_tracks = []
    for score, track in recs[:config['max_num_new_songs']]:
        if score>0.5:
            # We are going in descending order, but still skip if a song is sub-standard
            new_tracks.append(track)
        else:
            break
    
    if len(new_tracks)!=0:
        new_playlists.append({'name':new_name, 'tracks':new_tracks})
        
#### Save the new playlists to outputs 
if not os.path.exists(config['out_dir']):
    os.makedirs(config['out_dir']) 
    
with open(os.path.join(config['out_dir'], 'new_playlists.json'), 'w') as out_file:
    json.dump({'playlists':new_playlists}, out_file, indent=4)
    
