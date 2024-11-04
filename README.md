# LLM-Manager

This repository contains AI-functionality an LLM-Manager that can ```curate new songs recommendations to build playlists```, ```manage a social media account``` and ```summarize notifications```.

**Core LLM-logic defining an ```LLM_Agent``` and ```LLM_Recommender``` have been packaged together for easy-reuse in ```LLM_Engine```.
More details can be found in the ```LLM_Engine README```**.

High-level AI-functionality has been compiled as three main scripts:

- playlist_curator.py
- social_agent.py
- summarize_notifications.py

Data is read from ```data``` and stored to ```outputs```. 

#### Dependencies
Dependencies are listed in ```requirements.txt```

#### Assumptions
It is assumed that some other code runs the server, calls these scripts as appropriate and further processes outputs.
Further, load-times for models on each script run are ignored - ideally we would like a ready-server with already loaded LLM-models in memory - but that functionality is left to future work.

#### Settings and Prompt-Engineering
Settings for each of these scripts can be configured in ```config```, and prompt-engineering can be done by updating the text-files in each script's correponding ```prompts``` directory.


#### Functionality

- Running ```python playlist_curator.py``` will scrape the top-charts of most popular music from the web and use the LLM_Recommender to recommend new music tracks based on existing playlists. New music playlists once created are stored to a 'new_playlists.json' file in outputs.

- Running ```python social_agent.py``` will get information from configured data-sources (that may include **fitness**, **purchase** and **music** information) and use the LLM_Agent to generate a new social-media post in the style of previous user-posts. The new post generated will be stored in a time-stamped 'auto_post.txt' file in outputs.

- Running ```python summarize_notifications.py``` will use the LLM_Agent to summarize all of today's notificatios so far and save the output in a time-stamped 'notifications_summary.txt' file in outputs. (Note: You can test this by changing the variable ```today``` in this script if you do not want to use the current system-date-time).


#### Insights During Development
- Masking padded-token outputs before averaging word-vecs in the LLM_Recommender makes recommendations much better (now done by default).

- Choosing an appropriate ```max_seq_len``` and ```temperature``` for sampling for the LLM_Agent for each task can change behaviour drastically : These choices have been set in each script after experimenting with a few samples, but alternatives can be explored.

- Providing data to LLM_Agent in the form of stringified JSON objects seems to work well.

- Asking LLM_Agent to summarize an empty list of notifications makes the model hallucinate. A safety check has been added to prevent this.
