import json
import torch
import math
import random
import numpy as np

from transformers import AutoTokenizer


def read_json(path):
    """

    :param path: resolute path for json files using json.dump function to construct
    :return: dict
    """
    with open(path, "r", encoding='utf-8') as f:
        data = json.load(f)
    return data


def write_json(path, data):
    with open(path, "w", encoding='utf-8') as f:
        json.dump(data, f)


def reads_json(path):
    with open(path, "r") as f:
        data = [json.loads(line) for line in f]
        return data


def pad_tensor(vec, pad, dim):
    """
        Pads a tensor with zeros according to arguments given

        Args:
            vec (Tensor): Tensor to pad
            pad (int): The total tensor size with pad
            dim (int): Dimension to pad

        Returns:
            padded_tensor (Tensor): A new tensor padded to 'pad' in dimension 'dim'

    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    # torch.zeros torch.float32
    padded_tensor = torch.cat([vec, torch.zeros(*pad_size)], dim=dim)
    return padded_tensor


class PadCollate:
    def __init__(self, img_dim=0, twitter_dim=1, dep_dim=2, label_dim=3, knowledge_dim=4, knowledge_dep_dim=5,
                 use_np=False, max_know_len=20, knwoledge_type=1):
        """
        Args:
            img_dim (int): dimension for the image bounding boxes
            embed_dim1 (int): dimension for the matching caption
            embed_dim2 (int): dimension for the non-matching caption
            type
        """

        self.img_dim = img_dim
        self.twitter = twitter_dim
        self.dep = dep_dim
        self.label_dim = label_dim

        self.use_np = use_np
        self.knowledge_dim = knowledge_dim
        self.knowledge_dep_dim = knowledge_dep_dim
        self.max_len_know = max_know_len
        self.knowledge_type = knwoledge_type
        # img, twitter, dep, label, knowledge
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

    def pad_collate(self, batch):
        """
            A variant of collate_fn that pads according to the longest sequence in a batch of sequences and forms the minibatch

            Args:
                batch: Tuple. (img, embed_txt, org_seq, org_dep, org_word_len, org_token_len, org_label, org_chunk)

            Returns:
                xs(batchsize,49, 768): Tensor.
                embed_batch1(batchsize,L,768): Tensor. L is the max token length of input captions. Padded
                org_seq(batchsize,number of word): chunk index in Bert token for original caption.
                mask_batch1: (N,max_word_length). Tensor. key_padding_mask for word(np) for original caption
                org_token_len(batchsize):list of token length in a minibatch for RNN in text encoder
                edge_cap1(batchsize,number of edges, 2): list of Tensor.
                gnn_mask_1(batchsize): Boolean Tensor for orginal caption.
                np_mask_1(batchsize,max_length1+1): mask during the importance of np and caption
        """

        xs = list(map(lambda t: t[self.img_dim].clone().detach(), batch))
        xs = torch.stack(xs)
        # 获取batch中 token caption 的最大长度

        twitters = list(map(lambda t: t[self.twitter], batch))
        token_lens = [len(twitter) for twitter in twitters]

        encoded_cap = self.tokenizer(twitters, is_split_into_words=True, return_tensors="pt", truncation=True,
                                     max_length=100, padding=True)
        knowledges = list(map(lambda t: t[self.knowledge_dim], batch))
        real_token_len = [len(knowledge) for knowledge in knowledges]
        if self.knowledge_type != 1:
            knowledges_ = []
            for sample in knowledges:
                for word in sample:
                    knowledges_.append([word])
            knowledges = knowledges_

        encoded_know = self.tokenizer(knowledges, is_split_into_words=True, return_tensors="pt", truncation=True,
                                          max_length=30, padding=True)
        # len = 1  fo knwoledge 2,3
        know_token_lens = [len(knowledge) for knowledge in knowledges]
        img_patch_lens = [len(img) for img in xs]

        know_word_spans = []
        know_word_lens = []  # for mask matrix
        for index_encode, len_token in enumerate(know_token_lens):
            word_span_ = []
            if len_token > self.max_len_know:
                len_token = self.max_len_know
            for i in range(len_token):
                word_span = encoded_know[index_encode].word_to_tokens(i)
                if word_span is not None:
                    # delete [CLS]
                    word_span_.append([word_span[0] - 1, word_span[1] - 1])
            know_word_spans.append(word_span_)
            know_word_lens.append(len(word_span_))
        word_spans = []
        word_len = []
        for index_encode, len_token in enumerate(token_lens):
            word_span_ = []
            for i in range(len_token):
                word_span = encoded_cap[index_encode].word_to_tokens(i)
                if word_span is not None:
                    # delete [CLS]
                    word_span_.append([word_span[0] - 1, word_span[1] - 1])
            word_spans.append(word_span_)
            word_len.append(len(word_span_))
        # make sure each knowledge is isometric
        if self.knowledge_type > 1:
            max_len_know = max(real_token_len)
        else:
            max_len_know = self.max_len_know

        # max_len_know = max(know_word_lens)
        max_len1 = max(word_len)
        max_len_img = max(img_patch_lens)
        # mask矩阵是相对于word token的  key_padding_mask for computing the importance of each word in txt_encoder and
        # interaction modules
        if self.knowledge_type == 1:
            mask_batch_know = construct_mask_text(know_word_lens, max_len_know)
        else:
            mask_batch_know = construct_mask_text(real_token_len, max_len_know)

        mask_batch_img = construct_mask_text(img_patch_lens, max_len_img)
        mask_batch1 = construct_mask_text(word_len, max_len1)

        deps_know = [x[self.knowledge_dep_dim] for x in batch]
        deps1 = [x[self.dep] for x in batch]

        deps_know_ = []
        deps1_ = []
        # to avoid index out of range
        for dep in deps_know:
            deps_know_.append([d for d in dep if d[0] < max_len_know and d[1] < max_len_know])
        for dep in deps1:
            deps1_.append([d for d in dep if d[0] < max_len1 and d[1] < max_len1])
        # chunk index

        if self.use_np:
            org_chunk = list(map(lambda t: torch.tensor(t[self.chunk], dtype=torch.long), batch))
        else:
            org_chunk = [torch.arange(i, dtype=torch.long) for i in word_len]


        labels = torch.tensor(list(map(lambda t: t[self.label_dim], batch)), dtype=torch.long)
        edge_cap1, gnn_mask_1, np_mask_1 = construct_edge_text(deps=deps1_, max_length=max_len1, use_np=self.use_np,
                                                               chunk=org_chunk)
        # np_mask_know in useless
        edge_cap_know, gnn_mask_know = construct_edge_know(deps=deps_know_)

        # for caption knowledge
        return xs, encoded_cap, word_spans, word_len, mask_batch1, edge_cap1, gnn_mask_1, np_mask_1, \
               labels, \
               encoded_know, know_word_spans, mask_batch_know, edge_cap_know, gnn_mask_know, mask_batch_img

    def __call__(self, batch):
        return self.pad_collate(batch)


class PadCollate_without_know:
    def __init__(self, img_dim=0, twitter_dim=1, dep_dim=2, label_dim=3, chunk_dim=4, use_np=False):
        """
        Args:
            img_dim (int): dimension for the image bounding boxes
            embed_dim1 (int): dimension for the matching caption
            embed_dim2 (int): dimension for the non-matching caption
            type
        """

        self.img_dim = img_dim
        self.twitter = twitter_dim
        self.dep = dep_dim
        self.label_dim = label_dim
        self.chunk = chunk_dim

        self.use_np = use_np
        # img, text_emb, text_seq, dep, word_len, token_len, label

        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

    def pad_collate(self, batch):
        """
            A variant of collate_fn that pads according to the longest sequence in a batch of sequences and forms the minibatch
            Args:
                batch: Tuple. (img, embed_txt, org_seq, org_dep, org_word_len, org_token_len, org_label, org_chunk)
            Returns:
                xs(batchsize,49, 768): Tensor.
                embed_batch1(batchsize,L,768): Tensor. L is the max token length of input captions. Padded
                org_seq(batchsize,number of word): chunk index in Bert token for original caption.
                mask_batch1: (N,max_word_length). Tensor. key_padding_mask for word(np) for original caption
                org_token_len(batchsize):list of token length in a minibatch for RNN in text encoder
                edge_cap1(batchsize,number of edges, 2): list of Tensor.
                gnn_mask_1(batchsize): Boolean Tensor for orginal caption.
                np_mask_1(batchsize,max_length1+1): mask during the importance of np and caption
        """

        xs = list(map(lambda t: t[self.img_dim].clone().detach(), batch))
        xs = torch.stack(xs)
        # 获取batch中 token caption 的最大长度

        twitters = list(map(lambda t: t[self.twitter], batch))
        token_lens = [len(twitter) for twitter in twitters]

        encoded_cap = self.tokenizer(twitters, is_split_into_words=True, return_tensors="pt", truncation=True,
                                     max_length=100, padding=True)

        word_spans = []
        word_len = []
        for index_encode, len_token in enumerate(token_lens):
            word_span_ = []
            for i in range(len_token):
                word_span = encoded_cap[index_encode].word_to_tokens(i)
                if word_span is not None:
                    # delete [CLS]
                    word_span_.append([word_span[0] - 1, word_span[1] - 1])
            word_spans.append(word_span_)
            word_len.append(len(word_span_))

        max_len1 = max(word_len)
        # mask矩阵是相对于word token的  key_padding_mask for computing the importance of each word in txt_encoder and
        # interaction modules
        mask_batch1 = construct_mask_text(word_len, max_len1)

        img_patch_lens = [len(img) for img in xs]
        max_len_img = max(img_patch_lens)
        mask_batch_img = construct_mask_text(img_patch_lens, max_len_img)

        deps1 = [x[self.dep] for x in batch]

        deps1_ = []
        # to avoid index out of range
        for dep in deps1:
            deps1_.append([d for d in dep if d[0] < max_len1 and d[1] < max_len1])
        # chunk index
        if self.use_np:
            org_chunk = list(map(lambda t: torch.tensor(t[self.chunk], dtype=torch.long), batch))
        else:
            org_chunk = [torch.arange(i, dtype=torch.long) for i in word_len]

        labels = torch.tensor(list(map(lambda t: t[self.label_dim], batch)), dtype=torch.long)
        edge_cap1, gnn_mask_1, np_mask_1 = construct_edge_text(deps=deps1_, max_length=max_len1, use_np=self.use_np,
                                                               chunk=org_chunk)

        # for image graph
        # attr_img = construct_edge_attr()
        return xs, encoded_cap, word_spans, word_len, mask_batch1, edge_cap1, gnn_mask_1, np_mask_1, labels, mask_batch_img

    def __call__(self, batch):
        return self.pad_collate(batch)


def construct_mask_text(seq_len, max_length):
    """

    Args:
        seq_len1(N): list of number of words in a caption without padding in a minibatch
        max_length: the dimension one of shape of embedding of captions of a batch

    Returns:
        mask(N,max_length): Boolean Tensor
    """
    # the realistic max length of sequence
    max_len = max(seq_len)
    if max_len <= max_length:
        mask = torch.stack(
            [torch.cat([torch.zeros(len, dtype=bool), torch.ones(max_length - len, dtype=bool)]) for len in seq_len])
    else:
        mask = torch.stack(
            [torch.cat([torch.zeros(len, dtype=bool),
                        torch.ones(max_length - len, dtype=bool)]) if len <= max_length else torch.zeros(max_length,
                                                                                                         dtype=bool) for
             len in seq_len])

    return mask


# edge边 构造方法 [CLS]
def construct_edge_text(deps, max_length, chunk=None, use_np=False):
    """

    Args:
        deps: list of dependencies of all captions in a minibatch
        chunk: use to confirm where
        max_length : the max length of word(np) length in a minibatch
        use_np:

    Returns:
        deps(N,2,num_edges): list of dependencies of all captions in a minibatch. with out self loop.
        gnn_mask(N): Tensor. If True, mask.
        np_mask(N,max_length+1): Tensor. If True, mask
    """
    dep_se = []
    gnn_mask = []
    np_mask = []
    if use_np:
        for i, dep in enumerate(deps):
            if len(dep) > 1 and len(chunk[i]) > 1:
                # dependency between word(np) and word(np)
                dep_np = [torch.tensor(dep, dtype=torch.long), torch.tensor(dep, dtype=torch.long)[:, [1, 0]]]
                dep_np = torch.cat(dep_np, dim=0).T.contiguous()
                gnn_mask.append(False)
                np_mask.append(True)
            else:
                dep_np = torch.tensor([])
                gnn_mask.append(True)
                np_mask.append(False)
            dep_se.append(dep_np)
    else:
        for i, dep in enumerate(deps):
            if len(dep) > 3 and len(chunk[i]) > 1:
                dep_np = [torch.tensor(dep, dtype=torch.long), torch.tensor(dep, dtype=torch.long)[:, [1, 0]]]
                gnn_mask.append(False)
                np_mask.append(True)
                dep_np = torch.cat(dep_np, dim=0).T.contiguous()
            else:
                dep_np = torch.tensor([])
                gnn_mask.append(True)
                np_mask.append(False)
            dep_se.append(dep_np.long())

    np_mask = torch.tensor(np_mask).unsqueeze(1)
    np_mask_ = [torch.tensor(
        [True] * max_length) if gnn_mask[i] else torch.tensor([True] * max_length).index_fill_(0, chunk_,
                                                                                               False).clone().detach()
                for i, chunk_ in enumerate(chunk)]
    np_mask_ = torch.stack(np_mask_)
    np_mask = torch.cat([np_mask_, np_mask], dim=1)
    gnn_mask = torch.tensor(gnn_mask)
    return dep_se, gnn_mask, np_mask


def construct_edge_know(deps):
    """

    Args:
        deps: list of dependencies of all captions in a minibatch
        chunk: use to confirm where
        max_length : the max length of word(np) length in a minibatch
        use_np:

    Returns:
        deps(N,2,num_edges): list of dependencies of all captions in a minibatch. with out self loop.
        gnn_mask(N): Tensor. If True, mask.
        np_mask(N,max_length+1): Tensor. If True, mask
    """
    dep_se = []
    gnn_mask = []
    for i, dep in enumerate(deps):
        if len(dep) > 1:
            dep_np = [torch.tensor(dep, dtype=torch.long), torch.tensor(dep, dtype=torch.long)[:, [1, 0]]]
            gnn_mask.append(False)
            dep_np = torch.cat(dep_np, dim=0).T.contiguous()
        else:
            dep_np = torch.tensor([])
            gnn_mask.append(True)
        dep_se.append(dep_np.long())
    gnn_mask = torch.tensor(gnn_mask)
    return dep_se, gnn_mask


def construct_edge_image(num_patches):
    """
    Args:
        num_patches: the patches of image (49)
    There are two kinds of construct method
    Returns:
        edge_image(2,num_edges): List. num_edges = num_boxes*num_boxes
    """
    # fully connected 构建方法
    edge_image = []
    # for i in range(num_patches):
    #     edge_image.append(torch.stack([torch.full([num_patches], i, dtype=torch.long),
    #                                    torch.arange(num_patches, dtype=torch.long)]))
    # edge_image = torch.cat(edge_image, dim=1)
    # remove self-loop
    p = math.sqrt(num_patches)
    for i in range(num_patches):
        for j in range(num_patches):
            if j == i:
                continue
            if math.fabs(i % p - j % p) <= 1 and math.fabs(i // p - j // p) <= 1:
                edge_image.append([i, j])
    edge_image = torch.tensor(edge_image, dtype=torch.long).T
    return edge_image


def construct_edge_attr(bboxes, imgs):
    """

    Args:
        bboxes(N,num_bboxes,4): the last dimension is (x1,y1) and (x2,y2) referring to start vertex and end vertex

    Returns:
        bboxes_festures(N, num_edges, num_edge_features) : List of Tensor. Size of Tensor is
        (num_edges, num_edge_features)

    """
    bboxes_festures = []
    for i, sample in enumerate(bboxes):
        s_img = imgs[i].size(0) * imgs[i].size(1)
        sample_feature = []
        for o in sample:
            bbox_feature = []
            wo = o[2] - o[0]
            ho = o[3] - o[1]
            so = wo * ho
            for s in sample:
                ws = s[2] - s[0]
                hs = s[3] - s[1]
                bbox_feature.append(
                    [(o[0] - s[0]) / ws, (o[1] - s[1]) / hs, math.log(wo / ws), math.log(ho / hs), so / s_img])
            sample_feature.append(torch.tensor(bbox_feature))
        # (N, bbox_number,bbox_number,5)
        bboxes_festures.append(torch.cat(sample_feature))
    return torch.tensor(bboxes_festures)


def seed_everything(seed: int):
    r"""Sets the seed for generating random numbers in PyTorch, numpy and
    Python.

    Args:
        seed (int): The desired seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
