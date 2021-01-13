# Personality Trait Detection with MTL

This is the implementation about [Multitask Learning for Emotion and Personality Detection](https://arxiv.org/abs/2101.02346).

## Environments
`python >= 3.6`
`pytorch>=1.6.0`

## Dataset

The data should be organized in the.py file as follows:

```python
'''["joy":0, "anger":1, "disgust":2, "shame":3, "guilt":4, "fear":5, "sadness":6]'''
dataset = [
	("During the period of falling in love, each time that we met and especially when we had not met for a long time.", 0),
	("When I was involved in a traffic accident.", 5)
]
```

## Embedding

The default embedding is [Glove](http://nlp.stanford.edu/data/glove.6B.zip "Glove").



## Running Code

`python main.py`

