import json

import torch 
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 


with open("LLM_Engine/config/recommender_config.json", "r") as config_file: 
    recommender_config = json.loads(config_file.read())

class LLM_Recommender():
    """
    An LLM-Based recommender that will take a list of candidate sentences, a list of profile sentences and 
    sort the candidate sentences by their similaity to the given profile.
    """
    def __init__(self):
        self.similarity = nn.CosineSimilarity(dim=1)

        self.device = recommender_config['device']
        self.model_ckpt = recommender_config['model_ckpt']
        self.max_seq_length = recommender_config['max_seq_length']

        self.model = AutoModelForCausalLM.from_pretrained(self.model_ckpt,
                                                          attn_implementation="eager",
                                                          torch_dtype="auto",
                                                          output_hidden_states=True,
                                                          trust_remote_code=True)

        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_ckpt)
        self.tokenizer.pad_token = self.tokenizer.eos_token


    def get_sent_embeddings(self, sents, avg_sents=False):
        batch_size = recommender_config['batch_size']
        
        i = 0
        sentence_embeddings = []
        while i<len(sents):
            batch_sent = sents[i:batch_size+i]
            
            batch_tokenized = self.tokenizer(batch_sent, return_tensors='pt', padding=True,
                                             truncation=True,
                                             max_length=self.max_seq_length)

            batch_tokenized = batch_tokenized.to(self.device)

            # Get the hidden states (word-vecs) output by the last transformer block 
            batch_outputs = self.model(**batch_tokenized).hidden_states[-1].detach()
        
            # Mask to ignore padding tokens
            batch_masked_word_embeddings = batch_outputs * batch_tokenized.attention_mask.unsqueeze(-1).float()
            batch_num_non_pad_tokens = batch_tokenized.attention_mask.sum(dim=1, keepdim=True).float()

            # Sum across word-vecs and normalize embeddings (considering only non-padding tokens) to get sent-embeddings 
            batch_sentence_embeddings = batch_masked_word_embeddings.sum(dim=1) / batch_num_non_pad_tokens

            # Update counter
            i += batch_size
            
            # Append to list
            sentence_embeddings.append(batch_sentence_embeddings)

        # Recombine batches
        sentence_embeddings = torch.vstack(sentence_embeddings)
            
        if avg_sents:
          # Average all sentence embeddings for a profile
          sentence_embeddings = sentence_embeddings.mean(dim=0)

        return sentence_embeddings


    def get_candidate_sim_scores(self, candidate_sents, profile_sents):
        """
        Compute sim scores between candidate and profile sentences
        """
        candidate_embeddings = self.get_sent_embeddings(candidate_sents)
        profile_embedding = self.get_sent_embeddings(profile_sents, avg_sents=True)

        return self.similarity(candidate_embeddings, profile_embedding)

    def sort_by_recommendation(self, candidate_sents, profile_sents):
        """
        Sort and return candidates by recommendation
        """
        # Get candidate similarity scores in relation to the profile
        sim_scores = self.get_candidate_sim_scores(candidate_sents, profile_sents).tolist()

        # Tag the candidate list with their sim scores
        sim_tagged_candidates = list(zip(sim_scores, candidate_sents))
        
        # Sort candidates by similarity to the profile
        sorted_candidates = list(reversed(sorted(sim_tagged_candidates, key=lambda x: x[0])))

        return sorted_candidates
