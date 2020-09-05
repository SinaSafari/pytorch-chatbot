import random
import json
import numpy as np
import nltk
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk_utils import bag_of_words, tokenize, stem
from nltk.stem.porter import PorterStemmer

# Module
class NeuralNet(nn.modules):
	"""
	for more info:
	https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module
	"""

	def __init__(self, input_size, hidden_size, num_classes):
		'''
		classes are the number of output neurons
		'''
		super(NeuralNet, self).__init__()
		self.l1 = nn.Linear(input_size, hidden_size)
		self.l2 = nn.Linear(hidden_size, hidden_size)
		self.l3 = nn.Linear(hidden_size, num_classes)
		self.relu = nn.ReLU() # activation function

	def forward(self, x):
		'''
		for last step we don't need to run activation function
		CrossEntropyLoss handle this part
		'''
		out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)

        return out


# The Porter stemming algorithm (or ‘Porter stemmer’) is a process for
# removing the commoner morphological and inflexional endings from words in English.
# more info: 
# https://tartarus.org/martin/PorterStemmer/
stemmer = PorterStemmer()

# just for the first time uncomment below code
#
# Punkt Sentence Tokenizer:
# This tokenizer divides a text into a list of sentences by using
# an unsupervised algorithm to build a model for 
# abbreviation words, collocations, and words that start sentences. 
# nltk.download('punkt')


def tokenize(sentence):
	"""
    split sentence into array of words/tokens
    a token can be a word or punctuation character, or number
    """
	return nltk.word_tokenize(sentence)


def stem(word):
	"""
    stemming = find the root form of the word
    examples:
    words = ["organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
	return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
	"""
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bag   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
	tokenized_sentence = [stem(w) for w in tokenized_sentence]
    np.zeros(len(all_words), dtype=np.float32)
    for isx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag



with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
# loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = tokenize(pattern)
        # add to our words list
        all_words.extend(w)
        # add to xy pair
        xy.append((w, tag))

# stem and lower each word
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# create training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters 
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)


class ChatDataset(Dataset):
	'''
	pytorch map-style dataset
	"represents a map from (possibly non-integral) indices/keys to data samples."
	'''

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

# checking for GPU and if it's available it runs on gpu otherwise run on CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
# for more info: 
# https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss
criterion = nn.CrossEntropyLoss()
# adam algorithm: "for first-order gradient-based optimization of 
# stochastic objective functions, 
# based on adaptive estimates of lower-order moments."
# for more info: https://arxiv.org/abs/1412.6980
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
# main training loop
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')

# save training data for further using in chat application
data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')