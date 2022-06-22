"""
    Class file that enlists models for extracting features from image
"""
import torch
from torch import nn
from pytorch_pretrained_vit import ViT

class ImageEncoder(nn.Module):
    def __init__(self, input_dim=768, inter_dim=500, output_dim=300):
        """
            Initializes the model to process bounding box features extracted by MaskRCNNExtractor
            Returns:
                None
        """
        super(ImageEncoder, self).__init__()
        # model_name = 'B_32'
        # self.vit = ViT(model_name, pretrained=True)
        # self.vit.__delattr__("fc")

        self.input_dim = input_dim
        self.inter_dim = inter_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(self.input_dim, self.inter_dim)
        self.fc2 = nn.Linear(self.inter_dim, self.output_dim)
        self.fc3 = nn.Linear(self.output_dim, 1)
        self.relu1 = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.norm = torch.nn.LayerNorm(self.output_dim)



    def forward(self, x, lam=1):
        """
            Function to compute forward pass of the ImageEncoder
        Args:
            x (Tensor): Bounding box features extracted from Mask R-CNN. Tensor of shape (N,K,2048,7,7)
            where N is the batch size and K is the bboxes number, features map of shape 2048 X 7 x 7 for each object
            edge_index(tensor, dtype=torch.long): Graph connectivity in COO format. Tensor of shape (2,num_edges).
            Because in general, the image graph is fully connected.
            edge_attr(tensor, dtype=torch.float) : Edge feture with shape (N, num_edges, num_features)
        Returns:
            x (Tensor): Processed bounding box features. Tensor of shape (N,K,output_dim), K=49
            pv: (N,K) Tensor, the importance of each visual object of image
            """
        # if len(x.shape)==4:
        #         # remove class token
        #     x = self.vit(x)[:, :-1, :]
        # x shape (N,K,output_dim)
        x = self.relu1(self.fc2(self.relu1(self.fc1(x))))
        # 后续要修改成batch
        # 需要放在cuda上
        x = self.norm(x)
        # softmax need to check the output of linear where overflow or underflow
        # (N,K,1)

        pv = self.fc3(x).squeeze()
        # (N, K)
        pv = self.softmax(pv*lam)

        return x, pv
