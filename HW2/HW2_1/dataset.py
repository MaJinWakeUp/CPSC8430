import os
import torch
from torch.utils.data import Dataset
import json
import numpy as np
import re

class Text2Embedding():
    def __init__(self, word_dict, max_length=50):
        self.word2id = json.load(open(word_dict, 'r'))
        self.max_length = max_length
        self.num_words = len(self.word2id)
        self.num_captions = 20
        self.id2word = {v: k for k, v in self.word2id.items()}

    def _parse_sentence(self, sentence):
        pattern = r'\w+|\S'
        tokens = re.findall(pattern, sentence.lower())
        tokens.insert(0, '<BOS>')
        tokens.append('<EOS>')
        return tokens

    def __call__(self, captions):
        embeddings = []
        for caption in captions:
            embedding = []
            tokens = self._parse_sentence(caption)
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]
            else:
                tokens += ['<PAD>'] * (self.max_length - len(tokens))

            for token in tokens:
                one_hot = self.word2id.get(token, self.word2id['<UNK>'])
                embedding.append(one_hot)
            embeddings.append(embedding)
        if len(embeddings) < self.num_captions:
            num_replicas = self.num_captions - len(embeddings)
            embeddings += [embeddings[i % len(embeddings)] for i in range(num_replicas)]
        embeddings = embeddings[:self.num_captions]
        return torch.LongTensor(embeddings)

class MyDataset(Dataset):
    def __init__(self, data_dir, split='training', pre_extracted=True, caption_transform=None):
        self.pre_extracted = pre_extracted
        self.split = split
        self.caption_transform = caption_transform
        if pre_extracted:
            self.data_folder = os.path.join(data_dir, split+'_data', 'feat')
        else:
            self.data_folder = os.path.join(data_dir, split+'_data', 'video')
        annotation_file = os.path.join(data_dir, split + '_label.json')
        self.data = json.load(open(annotation_file, 'r'))

    def __len__(self):
        return len(self.data)
    
    def load_video(self, video_path):
        # load video
        pass

    def __getitem__(self, idx):
        id, caption = self.data[idx]['id'], self.data[idx]['caption']
        if self.pre_extracted:
            vfeat = torch.from_numpy(np.load(os.path.join(self.data_folder, id + '.npy'))).float()
        else:
            vfeat = self.load_video(os.path.join(self.data_folder, id))
        if self.caption_transform:
            caption = self.caption_transform(caption)
        return vfeat, caption

if __name__ == '__main__':
    data_dir = './MLDS_hw2_1_data'
    embed = Text2Embedding('word_dict.json', max_length=50)
    dataset = MyDataset(data_dir, 'training', pre_extracted=True, caption_transform=embed)
    print(len(dataset))
    for i in range(10):
        vfeat, caption = dataset[i]
        print(vfeat.shape, caption.shape)
