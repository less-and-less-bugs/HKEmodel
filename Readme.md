# Introduction

Official implementation of **Towards Multi-Modal Sarcasm Detection via Hierarchical Congruity Modeling with Knowledge Enhancement**

# Run Code

To run our code, please replace paths of datasets in `main function` of *train.py* using your paths first. Also, your can unzip `twitter.zip` (dataset file)  and place it in sarcasm project to run code immediately. In addition, you can generate image embeddings, dependency and three kinds of knowledge using `data_process.ipynb`.

Then you need to specify path to `parameter.json` for *train.py*,  which contain hyperparamters for our model.  Concretely, two parameter files in our project to recover performance reported in our paper, including `parameter.json` with knowledge enhancement and `parameter_without_know.json` without knowledge enhancement.  However, you may need to tune the parameters for your machine.  Moreover,  please use one-layer MLP for text-image branch without knowledge and  two-layer MLP for text-image branch with knowledge at the final classification layer. That's because text-knowledge branch is one optimization shortcut so we need more complicated classification layer to mitigate this problem.

At last,  you can run the below code:

```bash
CUDA_VISIBLE_DEVICES=1 python train.py
```



For experiment, please refer to requirements. txt and we only list the core packages.

For dataset,  **as the anonymous rule,** we will upload the `twitter.zip`  **afterwards.**

# CheckList 

We perform our experimetns on 24-GB 3090Ti with nearly one hour for one run.

Total Params: 112540942

We take the average results of  multiple runs for reports.

# Citation

If you find this repo useful for your research, please consider citing the paper.

Thanks for dataset from https://github.com/headacheboy/data-of-multimodal-sarcasm-detection

Thanks for Vit model from https://github.com/lukemelas/PyTorch-Pretrained-ViT

Thanks for Clipcap model from https://github.com/rmokady/CLIP_prefix_caption

