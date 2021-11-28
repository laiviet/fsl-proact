# ed-fsl
Few-shot learning for event detection


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

# Run MetaOptNet models:

```
python dc.py <with arguments>
```


