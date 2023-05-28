## Persona Aware Response Generation

### Getting started : Preparing data
 - Install all requirements listed in `requirements.txt`.
 - Check arguments in `data/save2json.py`. Prepare data.

```shell
cd data
python save2json.py --subset train --encdec bart --pretrained_name facebook/bart-base --encrep first
python save2json.py --subset valid --encdec bart --pretrained_name facebook/bart-base --encrep first
cd ..
```

### How to train?
 - Check all arguments needed in `main.py`, then run it
```shell
python3 main.py --save_name models/trialRun --batch_size 128 --encdec bart --pretrained_name facebook/bart-base --encrep first
```
 - It will save the model to `models/trialRun`.

### How to generate?
 - Check all arguments needed in `evaluate.py`, these aer generation parameters.

#### You can read on controlling sampling methods in [How to Generate](https://huggingface.co/blog/how-to-generate)

 - It will generate responses from the save and save them comprehensively in `generation_results/` in a json, where you will get corresponding scores - per sentence and averaged.

### To-do list
 - [ ] Adj for other types of graphs
 - [ ] T5
 - [x] Metrics and evaluate script.
 - [x] update dataloader fields to match model forward
 - [x] modify main.py
 - [x] Save only trainable parameters?
 - [x] Flexibility to switch Graph types : adj + args