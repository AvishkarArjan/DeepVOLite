# Lets not mess this up
This project is my second try at replicating the results of the paper [DeepVO](https://arxiv.org/abs/1709.08429). The first attempt was so useless that I had to abandon weeks of coding and analysis altogether. 

CURRENT UPDATE - So trying to reduce the number of parameters randomly within the model causes blunders. The least MSE loss i was able to get (training on my cpu) was around 0.10 - which is ridiculous in this case. I might have to try some more things to try to create a lighter version, maybe a different normalization . Idk. 

## Differences from Original paper
The original model architecture mentioned in the paper 

I have limited computing resources - thus I drastically reduced the parameters by omitting out several layers from the model - to atleast me able to train the model. Based on some [research](https://stackoverflow.com/questions/69769574/how-to-efficiently-use-memory-in-google-colab), the max parameters of a model than can be trained free on Google Colab is under 1 million. 

## Some resources that may or may not have contributed to the development of this project:
https://github.com/thedavekwon/DeepVO

