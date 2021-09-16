
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
import flask
import os
import json
from collections import Counter
import torch
from random import choice
from IPython.display import clear_output
from torch import nn
from PIL import Image
import torch.nn.functional as F
import pickle 
import warnings
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models.inception import Inception3
from torch.utils.model_zoo import load_url

class BeheadedInception3(Inception3):
    """ Like torchvision.models.inception.Inception3 but the head goes separately """
    
    def forward(self, x):
        if self.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        else: warn("Input isn't transformed")
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x_for_attn = x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        x_for_capt = x = x.view(x.size(0), -1)
        # 2048
        x = self.fc(x)
        # 1000 (num_classes)
        return x_for_attn, x_for_capt, x
    

def beheaded_inception_v3(transform_input=True):
    model= BeheadedInception3(transform_input=transform_input)
    inception_url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
    model.load_state_dict(load_url(inception_url))
    return model


import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
import json
from collections import Counter
import torch
from random import choice
from IPython.display import clear_output
from torch import nn
import torch.nn.functional as F

n_tokens = 10403

class CaptionNet(nn.Module):
    
  def __init__(self, n_tokens=n_tokens, emb_size=128, lstm_units=256, cnn_feature_size=2048):
        """ A recurrent 'head' network for image captioning. . """
        super().__init__()
        
        # a layer that converts conv features to initial_h (h_0) and initial_c (c_0)
        self.cnn_to_h0 = nn.Linear(cnn_feature_size, lstm_units)
        self.cnn_to_c0 = nn.Linear(cnn_feature_size, lstm_units)
        self.lstm_units = lstm_units
        # create embedding for input words. Use the parameters (e.g. emb_size).
        self.embedding = nn.Embedding(n_tokens, emb_size)

        # lstm: create a recurrent core of your network.
        self.LSTM = nn.LSTM(emb_size,lstm_units,num_layers = 1,batch_first=True)
            
        # create logits: linear layer that takes lstm hidden state as input and computes one number per token
        
        self.linear = nn.Linear(lstm_units, n_tokens)
        
  def forward(self, image_vectors, captions_ix):
        """ 
        Apply the network in training mode. 
        :param image_vectors: torch tensor containing inception vectors. shape: [batch, cnn_feature_size]
        :param captions_ix: torch tensor containing captions as matrix. shape: [batch, word_i]. 
            padded with pad_ix
        :returns: logits for next token at each tick, shape: [batch, word_i, n_tokens]
        """

        self.LSTM.flatten_parameters()

        initial_cell = self.cnn_to_c0(image_vectors)
        initial_hid = self.cnn_to_h0(image_vectors)
        
        # compute embeddings for captions_ix
        x_emb = self.embedding(captions_ix)

        output, (hn, cn) = self.LSTM(x_emb, (initial_cell[None], initial_hid[None]))
        # compute logits from lstm_out
        output = self.linear(output)
        return output

warnings.filterwarnings("ignore")

with open('word_to_index.pickle', 'rb') as fp:
    word_to_index = pickle.load(fp)

n_tokens =10403

vocab = [k for k, v in word_to_index.items()]

eos_ix = word_to_index['#END#']
unk_ix = word_to_index['#UNK#']
pad_ix = word_to_index['#PAD#']

def as_matrix(sequences, max_len=None):
    """ Convert a list of tokens into a matrix with padding """
    max_len = max_len or max(map(len,sequences))
    
    matrix = np.zeros((len(sequences), max_len), dtype='int32') + pad_ix
    for i,seq in enumerate(sequences):
        row_ix = [word_to_index.get(word, unk_ix) for word in seq[:max_len]]
        matrix[i, :len(row_ix)] = row_ix
    
    return matrix


DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

inception = beheaded_inception_v3().eval()

def generate_caption(image, network, caption_prefix = ('#START#',), t=1, sample=True, max_len=100):
    network = network.cpu().eval()

    assert isinstance(image, np.ndarray) and np.max(image) <= 1\
           and np.min(image) >= 0 and image.shape[-1] == 3
    
    image = torch.tensor(image.transpose([2, 0, 1]), dtype=torch.float32)
    
    vectors_8x8, vectors_neck, logits = inception(image[None])
    caption_prefix = list(caption_prefix)
    
    for _ in range(max_len):
        
        prefix_ix = as_matrix([caption_prefix])
        prefix_ix = torch.tensor(prefix_ix, dtype=torch.int64)
        next_word_logits = network.forward(vectors_neck, prefix_ix)[0, -1]
        next_word_probs = F.softmax(next_word_logits, -1).detach().numpy()
        
        assert len(next_word_probs.shape) == 1, 'probs must be one-dimensional'
        next_word_probs = next_word_probs ** t / np.sum(next_word_probs ** t) # apply temperature

        if sample:
            next_word = np.random.choice(vocab, p=next_word_probs) 
        else:
            next_word = vocab[np.argmax(next_word_probs)]

        caption_prefix.append(next_word)

        if next_word == '#END#':
            break

    return ' '.join(caption_prefix[1:-1])


restored_network = CaptionNet(n_tokens)
restored_network.load_state_dict(torch.load('result.bin', map_location=torch.device('cpu')))

#get image from url
app = flask.Flask(__name__)

@app.route('/')
def hello_world():
    return flask.render_template('index.html')


@app.route('/predict', methods=["POST"])
def transform_view():
    path_to_file = flask.request.files.get('picture')
    if not path_to_file:
        return "No file"
    #img = Image.open(path_to_file)
    img = plt.imread(path_to_file)
    
    img = resize(img, (299, 299))
    
    output = generate_caption(img, restored_network, t=5.)
    for i in range(5):
        print(generate_caption(img, restored_network, t=5.))
    response = flask.make_response(output)
    return flask.render_template("answer.html", output=output.split(' '))
    
if __name__ == "__main__":
    app.run(debug=False)