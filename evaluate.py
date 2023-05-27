import os
import json
import argparse

from tqdm import tqdm

from model import BartForPersonaAwareGeneration
from utils import PersonaDataset
from modules.evaluations import similariry_score, score_rouge, bleu_score

import torch
import torch.nn.functional as f
from torch.utils.data import DataLoader

from transformers import BartTokenizer
from sentence_transformers import SentenceTransformer


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIMILARITY_MODEL = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').cuda()


def parse_args() :
    parser = argparse.ArgumentParser()

    # model params, ignore
    parser.add_argument("--model", type = str, required = True)
    parser.add_argument("--max_length", type = int, default = 32)
    parser.add_argument("--encdec", type = str, default = "bart")
    parser.add_argument("--pretrained_name", type = str, default = "facebook/bart-base")
    parser.add_argument("--graph_type", type = str, default = "paperGCN")
    parser.add_argument("--encrep", type = str, default = "first",
                        help = "can be 'mean' or 'first'")
    parser.add_argument("--num_heads", type = int, default = 1)
    parser.add_argument("--max_conv_length", type = int, default = 50)
    parser.add_argument("--dropout",type = float, default = 0.25)

    # output file name
    parser.add_argument("--output_json_name", type = str, required = True) 
    parser.add_argument("--batch_size", type = int, default = 8)
    
    # greedy or sampling
    parser.add_argument("--do_sample", type = bool, default = False)

    # greedy decoding
    parser.add_argument("--num_beams", type = int, default = 1)

    # sampling methods
    parser.add_argument("--top_k", type = int, default = 50)
    parser.add_argument("--temperature", type = float, default = 1.0)
    parser.add_argument("--top_p", type = float, default = 1.0)

    # common
    parser.add_argument("--no_repeat_ngram_size", type = int, default = 0)
    parser.add_argument("--early_stopping", type = bool, default = True)
    parser.add_argument("--max_gen_length", type = int, default = 50)


    args = parser.parse_args()
    return args


def process_sample(sample) :
    conversation = {
        "conv_id" : sample["conv_id"],
        "personality_person2" : sample["personality_person2"],
        "dialog" : [d["text"] for d in sample["dialog"][ : -1]]
    }

    return conversation, sample["dialog"][-1]["text"]


def compute_metrics(logits, labels, generated, ground) :
    metrics = {
        "rougeL" : score_rouge(generated, ground),

        "bleu_1gram" : bleu_score(ground, generated, 1),
        "bleu_2gram" : bleu_score(ground, generated, 2),
        "bleu_3gram" : bleu_score(ground, generated, 3),
        "bleu_4gram" : bleu_score(ground, generated, 4),

        "sentence_similarity" : similariry_score(generated, ground, SIMILARITY_MODEL),

        "perplexity" : torch.exp(f.cross_entropy(logits, labels)).item()
    }

    return metrics


@torch.no_grad()
def generate(model, data, tokenizer, args) :
    # populate this
    results = {
        "generation_config" : {key : value for key, value in vars(args).items()},
        "averaged" : {},
        "results" : []
    }

    idx = 0
    with open("data/personachat_data.json", "r", encoding = "utf-8") as fi :
        reader = json.load(fi)["valid"]

    for batch in tqdm(data) :
        batch = {k : v.to(DEVICE) for k, v in batch.items()}

        outputs = model(
            conv_cls = batch["conv_cls"], persona_cls = batch["persona_cls"], 
            adj = batch["adj"], conv_mask = batch["conv_mask"],
            encoder_hidden_states = batch["encoder_hidden_states"],
            attention_mask = batch["attention_mask"],
            decoder_input_ids = batch["decoder_input_ids"],
            decoder_attention_mask = batch["decoder_attention_mask"],
            labels = batch["labels"]
            )
        
        encoder_hidden_states = outputs.encoder_last_hidden_state
        logits = outputs.logits

        # add other args here
        outputs = model.generate(
            inputs_embeds = encoder_hidden_states, 
            attention_mask = batch["attention_mask"],
            do_sample = args.do_sample, max_length = args.max_gen_length,
            early_stopping = args.early_stopping, 
            no_repeat_ngram_size = args.no_repeat_ngram_size,
            top_p = args.top_p, top_k = args.top_k,
            temperature = args.temperature,
            num_beams = args.num_beams
            )
        
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens = True)
        
        samples_here = batch["conv_mask"].shape[0]
        for i in range(samples_here) :
            # fail safe
            if not any(c.isalnum() for c in outputs[i]) :
                outputs[i] += "am"

            conversation, last = process_sample(reader[idx])
            
            conversation["next_response"] = outputs[i].strip()
            conversation["metrics"] = compute_metrics(logits[i], batch["labels"][i], 
                                                      outputs[i].strip(), last.strip())

            results["results"].append(conversation)
            idx += 1
        

    r = 0
    b1, b2, b3, b4 = 0, 0, 0, 0
    p = 0
    L = len(results["results"])
    for d in range(results["results"]) :
        r += d["metrics"]["rougeL"]
        b1 += d["metrics"]["bleu_1gram"]
        b2 += d["metrics"]["bleu_2gram"]
        b3 += d["metrics"]["bleu_3gram"]
        b4 += d["metrics"]["bleu_4gram"]
        p += d["metrics"]["perplexity"]
        
    results["averaged"]["rougeL"] = r / L
    results["averaged"]["bleu_1gram"] = b1 / L
    results["averaged"]["bleu_2gram"] = b2 / L
    results["averaged"]["bleu_3gram"] = b3 / L
    results["averaged"]["bleu_4gram"] = b4 / L
    results["averaged"]["perplexity"] = p / L

    with open("generation_results/" + args.model + "/" + args.output_json_name, "w", encoding = "utf-8") as fi :
        json.dump(results, fi, indent = 4)
    print("Results saved.")

    return


if __name__ == "__main__" :
    args = parse_args()

    # make dir for results
    if not os.path.isdir("generation_results") :
        os.mkdir("generation_results")
    if not os.path.isdir("generation_results/" + args.model) :
        os.mkdir("generation_results/" + args.model)

    # load data
    data = PersonaDataset(args, subset = "valid")
    data = DataLoader(data, batch_size = args.batch_size, shuffle = False)

    # load model
    model = BartForPersonaAwareGeneration.from_pretrained(args.model, args)
    model.to(DEVICE)
    model.eval()

    tokenizer = BartTokenizer.from_pretrained(args.pretrained_name)

    generate(model, data, tokenizer, args)
