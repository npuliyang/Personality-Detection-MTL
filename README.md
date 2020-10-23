# Personality Trait Detection with MTL

This is the implementation about the multi-task learning with personality trait detection and emotion detection.  

The code can be run directly after processing the data in the format as follows:

`##["joy":0, "anger":1, "disgust":2, "shame":3, "guilt":4, "fear":5, "sadness":6]
dataset = [
	("During the period of falling in love, each time that we met and especially when we had not met for a long time.", 0),
	("When I was involved in a traffic accident.", 5)
]`

## Requirement

`pytorch>=1.6.0`

## Runing Code

`python main.py --learning_rate 1e-4`