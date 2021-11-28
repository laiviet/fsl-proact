# ed-fsl
This is an official repo for papers:

[Learning Prototype Representations Across Few-Shot Tasks for Event Detection](https://aclanthology.org/2021.emnlp-main.427.pdf)

[Extensively Matching for Few-shot Learning Event Detection](https://aclanthology.org/2020.nuse-1.5.pdf)
# Prepare data

An event in json format has the following attributes:

preprocess/rams_utils.py generate a list of positive training example 



 ```
    train.json:
    
    event = {
                'id': '{}#{}'.format(doc_id, trigger_id),
                'token': tokens,    # List of tokens
                'trigger': [trigger_index_start, trigger_index_end],
                'label': label,
                'argument': arguments
            }
 ```
 preprocess/negative.py generates negative examples from positive examples
 
 ```
 train.json -> train.negative.json
 ```
 
 preprocess/graph.py run tokenizer, dependency parser and save to .parse file
 
 ```
 train.json -> train.parse.json
 train.negative.json -> train.negative.parse.json

 ```
 
 preprocess/prune.py prune dependency tree and save to .prune file
 ```
 train.parse.json -> train.prune.json
 train.negative.parse.json -> train.negative.prune.json
 ```
 
 preprocess/tokenizer.py run BERT tokenizer with ``bert-base-cased`` as BERT version
 
 ```
 train.prune.json -> train.bert-base-cased.json
 train.negative.prune.json -> train.negative.bert-base-cased.json
 ```

# Run most of the FSL model:

```
python fsl.py --dataset rams -n 5 -k 5 --encoder bertmlp --model proto
```

# Run ProAct model

```
python melr.py --dataset rams -n 5 -k 5 --encoder bertmlp --model melr
```

# Citatioin:

```
@inproceedings{lai2021learning,
  title={Learning Prototype Representations Across Few-Shot Tasks for Event Detection},
  author={Lai, Viet and Dernoncourt, Franck and Nguyen, Thien Huu},
  booktitle={Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
  pages={5270--5277},
  year={2021}
}
```

```
@inproceedings{lai2020extensively,
  title={Extensively Matching for Few-shot Learning Event Detection},
  author={Lai, Viet Dac and Nguyen, Thien Huu and Dernoncourt, Franck},
  booktitle={Proceedings of the First Joint Workshop on Narrative Understanding, Storylines, and Events},
  pages={38--45},
  year={2020}
}
```
