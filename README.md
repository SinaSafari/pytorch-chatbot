# Simple Chat bot - coffee shop
> using PyTorch and NLTK

### The NLP preprocessing pipeline
- Tokenize
- lower + stem
- exlude punctuation characters
- generate bag of words

***
### instruction
1. for training the neural network: (this should be done at first time)
```sh
python train.py
```
2. for chatting:
```sh
python chat.py
```

or you can use jupyter notebook

*** 
### Comments
- about [pytorch modules](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)
- The Porter stemming algorithm (or ‘Porter stemmer’) is a process for removing the commoner morphological and inflexional endings from words in English. [more info](https://tartarus.org/martin/PorterStemmer/)
- Punkt Sentence Tokenizer: This tokenizer divides a text into a list of sentences by using an unsupervised algorithm to build a model for abbreviation words, collocations, and words that start sentences.
- nltk tokenize: split sentence into array of words/tokens a token can be a word or punctuation character, or number
- nltk stem: stemming = find the root form of the word examples:
```py
words = ["organize", "organizes", "organizing"]
words = [stem(w) for w in words]
# return: ["organ", "organ", "organ"]
```
- bag of word: return bag of words array: 1 for each known word that exists in the sentence, 0 otherwise example:
```py
sentence = ["hello", "how", "are", "you"]
words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
bag   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
```
- Pytorch Dataset: map-style dataset: "represents a map from (possibly non-integral) indices/keys to data samples."
- Pytorch ```nn.CrossEntropyLoss```:[more info](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss)
- Adam Algorithm: "for first-order gradient-based optimization of stochastic objective functions, based on adaptive estimates of lower-order moments." [more info](https://arxiv.org/abs/1412.6980)


***
### Libraries
- [PyTorch](https://pytorch.org/)
- [NumPy](https://numpy.org/)
- [NLTK](https://www.nltk.org) (Natural Language Processing Toolkit)