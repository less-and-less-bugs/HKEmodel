import torch.nn as nn
import torch.nn.functional as F


class MCO(nn.Module):
    def __init__(self, input_size=300, nhead=6, dim_feedforward=600, dropout=0.1):
        super(MCO, self).__init__()
        self.co_att = nn.MultiheadAttention(input_size, nhead, dropout=dropout)
        self.linear1 = nn.Linear(input_size, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, input_size)
        self.norm1 = nn.LayerNorm(input_size)
        self.norm2 = nn.LayerNorm(input_size)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(MCO, self).__setstate__(state)

    def forward(self, tgt, src, src_key_padding_mask=None):
        """
        Args:
            tgt(L, N, E) : query matrix in MultiAttention. It's a Tensor.
            src(S, N, E) : Key, Value in MultiAttention. It's a Tensor.
            src_key_padding_mask(N,S) : if provided, specified padding elements in the key will
            be ignored by the attention. When given a binary mask and a value is True,
            the corresponding value on the attention layer will be ignored. When given
            a byte mask and a value is non-zero, the corresponding value on the attention
            layer will be ignored

        Shape for inputs:
            tgt(L, N, E)
        Returns:
            tgt(L,N,E): output of co-attention

        """
        if src_key_padding_mask is not None:
            # the tgt is image
            tgt2 = self.co_att(tgt, src, src, key_padding_mask=src_key_padding_mask)[0]
        else:
            # the tgt is text
            assert src_key_padding_mask is None, "The src has src_padding_mask but it's a image batch"
            tgt2 = self.co_att(tgt, src, src)[0]

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        return tgt


class BMCO(nn.Module):
    def __init__(self, input_size=300, nhead=6, dim_feedforward=600, dropout=0.2, type_mco=0):
        super(BMCO, self).__init__()

        self.input_size = input_size
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        self.type = type_mco

        if self.type == 0:
            self.cro_img = MCO(input_size=self.input_size, nhead=self.nhead, dim_feedforward=self.dim_feedforward,
                               dropout=self.dropout)
            self.cro_txt = MCO(input_size=input_size, nhead=self.nhead, dim_feedforward=self.dim_feedforward,
                               dropout=self.dropout)

        elif self.type == 1:
            self.cro_txt = MCO(input_size=input_size, nhead=self.nhead, dim_feedforward=self.dim_feedforward,
                               dropout=self.dropout)
        elif self.type == 2:
            self.cro_img = MCO(input_size=self.input_size, nhead=self.nhead, dim_feedforward=self.dim_feedforward,
                               dropout=self.dropout)

    def forward(self, images, texts, key_padding_mask, key_padding_mask_img=None):
        if self.type == 0:
            # means it is knowledge embedding
            # if key_padding_mask_img is not None:
            #     key_padding_mask_img = key_padding_mask_img.cuda()
            imgs1 = self.cro_img.forward(images, texts, src_key_padding_mask=key_padding_mask.cuda())
            texts1 = self.cro_txt.forward(texts, images, src_key_padding_mask=key_padding_mask_img.cuda())
        elif self.type == 1:
            # image key,value text query
            # if key_padding_mask_img is not None:
            #     key_padding_mask_img.cuda()
            texts1 = self.cro_txt.forward(texts, images, src_key_padding_mask=key_padding_mask_img.cuda())
            imgs1 = images
        elif self.type == 2:
            # text key,value text
            imgs1 = self.cro_img.forward(images, texts, src_key_padding_mask=key_padding_mask.cuda())
            texts1 = texts
        else:
            exit()
        return imgs1, texts1


class CroModality(nn.Module):
    def __init__(self, input_size=300, nhead=6, dim_feedforward=600, dropout=0.2, cro_layer=1, type_bmco=0):
        super(CroModality, self).__init__()

        self.input_size = input_size
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        self.cro_layer = cro_layer

        self.bmco_type = type_bmco

        self.cro_att = nn.ModuleList(
            [BMCO(input_size=self.input_size, nhead=self.nhead, dim_feedforward=self.dim_feedforward,
                  dropout=self.dropout, type_mco=self.bmco_type) for i in range(self.cro_layer)])

    def forward(self, images, texts, key_padding_mask, key_padding_mask_img=None):
        """
        Args:
            images(N,K,D): Tensor.
            texts(N,L,D): Tensor.
            key_padding_mask(N,L): Tensor. This is the same as text encoder.
        Returns:
            imgs2(N,): (N,K,D)Tensor.
            texts2(S,N,E): (N,L,D)Tensor.
        """
        images = images.permute(1, 0, 2)
        texts = texts.permute(1, 0, 2)
        # word np padding mask

        for i in range(self.cro_layer):
            images, texts = self.cro_att[i](images, texts, key_padding_mask, key_padding_mask_img=key_padding_mask_img)

        images = images.permute(1, 0, 2)
        texts = texts.permute(1, 0, 2)

        return images, texts
