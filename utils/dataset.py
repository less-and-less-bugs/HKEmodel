import torch
from torch.utils.data import Dataset
import json

class BaseSet(Dataset):
    def __init__(self, type="train", max_length=100, text_path=None, use_np=False, img_path=None, knowledge=0):
        """
        Args:
            type: "train","val","test"
            max_length: the max_lenth for bert embedding
            text_path: path to annotation file
            img_path: path to img embedding. Resnet152(,2048), Vit B_32(,768), Vit L_32(, 1024)
            use_np: True or False, whether use noun phrase as relation matching node. It is useless in this paper.
            img_path:
            knowledge: 1 caption, 2 ANP, 3 attribute, 0 not use knowledge
        """
        self.type = type  # train, val, test
        self.max_length = max_length
        self.text_path = text_path
        self.img_path = img_path
        self.use_np = use_np
        with open(self.text_path) as f:
            self.dataset = json.load(f)
        self.img_set = torch.load(self.img_path)
        self.knowledge = int(knowledge)

    def __getitem__(self, index):
        """

        Args:
            index:

        Returns:
            img: (49, 768). Tensor.
            text_emb: (token_len, 758). Tensor
            text_seq: (word_len). List.
            dep: List.
            word_len: Int.
            token_len: Int
            label: Int
            chunk_index: li

        """
        sample = self.dataset[index]

        # for val and test dataset, the sample[2] is hashtag label
        if self.type == "train":
            label = sample[2]
            text = sample[3]
        else:
            # label =sample[2] hashtag label
            label = sample[3]
            text = sample[4]
        # useless in this project
        if self.use_np:
            twitter = text["chunk_cap"]
            dep = text["chunk_dep"]
            chunk_index = text["chunk_index"]
        else:
            twitter = text["token_cap"]
            dep = text["token_dep"]

        img = self.img_set[index]
        if self.knowledge == 0:
            return img, twitter, dep, label

        knowledge = sample[-self.knowledge]
        # caption
        if self.knowledge == 1:
            knowledge_token = knowledge["token_cap"]
            knowledge_dep = knowledge["token_dep"]
        else:
            knowledge_token = knowledge
            knowledge_dep = []

        return img, twitter, dep, label, knowledge_token, knowledge_dep

    def __len__(self):
        """
            Returns length of the dataset
        """
        return len(self.dataset)

