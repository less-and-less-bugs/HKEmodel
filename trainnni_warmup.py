import argparse
from tqdm import tqdm
import os
import logging
from torch.utils.data import DataLoader
# from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.scheduler import ReduceLROnPlateau
import torch.optim as optim
import torch
from torch.nn import CrossEntropyLoss
import numpy as np

from model import KEHModel
from utils.data_utils import construct_edge_image
from utils.dataset import BaseSet
from utils.compute_scores import get_metrics, get_four_metrics
from utils.data_utils import PadCollate
from utils.data_utils import seed_everything
import json
import re

import nni

os.environ["TOKENIZERS_PARALLELISM"] = "true"
torch.multiprocessing.set_sharing_strategy('file_system')
logger = logging.getLogger('KEHN_AutoML')
# os.environ['CUDA_VISIBLE_DEVICES'] = "2"


def train_model(epoch, train_loader,model,cross_entropy_loss,optimizer,img_edge_index):
    """
        Performs one training epoch and updates the weight of the current model

        Args:
            train_loader:
            optimizer:
            epoch(int): Current epoch number

        Returns:
            None
    """
    train_loss = 0.
    total = 0.
    model.train()
    predict = []
    real_label = []
    # Training loop

    for batch_idx, (img_batch, embed_batch1, org_seq, org_word_len, org_token_len, mask_batch1,
       edge_cap1, gnn_mask_1, np_mask_1, labels) in enumerate(tqdm(train_loader)):
        batch = len(img_batch)
        with torch.set_grad_enabled(True):
            y = model(imgs=img_batch.cuda(), texts=embed_batch1.cuda(), mask_batch=mask_batch1.cuda(), img_edge_index=img_edge_index,
                      t1_token_length=org_token_len, t1_word_seq=org_seq, txt_edge_index=edge_cap1, gnn_mask=gnn_mask_1.cuda(),
                      np_mask=np_mask_1.cuda(), img_edge_attr=None)

            loss = cross_entropy_loss(y,labels.cuda())
            loss.backward()
            train_loss += float(loss.detach().item())
            optimizer.step()
            optimizer.zero_grad()  # clear gradients for this training step
        predict = predict + get_metrics(y.cpu())
        real_label = real_label + labels.cpu().numpy().tolist()
        total += batch
        torch.cuda.empty_cache()
        del img_batch, embed_batch1
    # Calculate loss and accuracy for current epoch

    acc,recall,precision,f1 = get_four_metrics(real_label, predict)
    logger.info('Train Epoch: {} Loss: {:.4f} Acc: {:.4f} Rec: {:.4f} Pre: {:.4f} F1: {:.4f}'.format(epoch, train_loss / len(train_loader), acc, recall,precision, f1))


def eval_validation_loss(val_loader,model,cross_entropy_loss,img_edge_index):
    """
        Computes validation loss on the saved model, useful to resume training for an already saved model
    """
    val_loss = 0.
    predict = []
    real_label = []
    model.eval()

    with torch.no_grad():
        # 在模型
        for batch_idx, (img_batch, embed_batch1, org_seq, org_word_len, org_token_len, mask_batch1,
                        edge_cap1, gnn_mask_1, np_mask_1, labels) in enumerate(tqdm(val_loader)):
            batch = len(img_batch)
            with torch.set_grad_enabled(True):
                y = model(imgs=img_batch.cuda(), texts=embed_batch1.cuda(), mask_batch=mask_batch1.cuda(), img_edge_index=img_edge_index,
                      t1_token_length=org_token_len, t1_word_seq=org_seq, txt_edge_index=edge_cap1, gnn_mask=gnn_mask_1.cuda(),
                      np_mask=np_mask_1.cuda(), img_edge_attr=None)

                loss = cross_entropy_loss(y, labels.cuda())
                val_loss += float(loss.clone().detach().item())
            predict = predict + get_metrics(y.cpu())
            real_label = real_label + labels.cpu().numpy().tolist()
            torch.cuda.empty_cache()
            del img_batch, embed_batch1
        acc, recall, precision, f1 = get_four_metrics(real_label, predict)
        print(' Val Avg loss: {:.4f} Acc: {:.4f} Rec: {:.4f} Pre: {:.4f} F1: {:.4f}'.format(val_loss / len(val_loader), acc, recall,
                                                                          precision, f1))
    return val_loss


def evaluate_model(epoch, val_loader,model,cross_entropy_loss,img_edge_index):
    """
        Performs one validation epoch and computes loss and accuracy on the validation set

        Args:
            model:
            epoch (int): Current epoch number

        Returns:
            val_loss (float): Average loss on the validation set
    """
    val_loss = 0.
    predict = []
    real_label = []
    model.eval()

    with torch.no_grad():
        # 在模型
        for batch_idx, (img_batch, embed_batch1, org_seq, org_word_len, org_token_len, mask_batch1,
                        edge_cap1, gnn_mask_1, np_mask_1, labels) in enumerate(tqdm(val_loader)):
            batch = len(img_batch)
            with torch.set_grad_enabled(True):
                y = model(imgs=img_batch.cuda(), texts=embed_batch1.cuda(), mask_batch=mask_batch1.cuda(), img_edge_index=img_edge_index,
                      t1_token_length=org_token_len, t1_word_seq=org_seq, txt_edge_index=edge_cap1, gnn_mask=gnn_mask_1.cuda(),
                      np_mask=np_mask_1.cuda(), img_edge_attr=None )

                loss = cross_entropy_loss(y, labels.cuda())
                val_loss += float(loss.clone().detach().item())
            predict = predict + get_metrics(y.cpu())
            real_label = real_label + labels.cpu().numpy().tolist()
            torch.cuda.empty_cache()
            del img_batch, embed_batch1
        acc, recall, precision, f1 = get_four_metrics(real_label, predict)
        logger.info('Val Epoch: {} Avg loss: {:.4f} Acc: {:.4f}  Rec: {:.4f} Pre: {:.4f} F1: {:.4f}'.format(epoch, val_loss / len(val_loader),
                                                                                                       acc, recall, precision, f1))
    return val_loss

def evaluate_model_test(epoch, test_loader,model,cross_entropy_loss,img_edge_index):
    """
        Performs one validation epoch and computes loss and accuracy on the validation set

        Args:
            epoch (int): Current epoch number
            test_loader:

        Returns:
            val_loss (float): Average loss on the validation set
    """
    test_loss = 0.
    predict = []
    real_label = []
    model.eval()

    with torch.no_grad():
        # 在模型
        for batch_idx, (img_batch, embed_batch1, org_seq, org_word_len, org_token_len, mask_batch1,
                        edge_cap1, gnn_mask_1, np_mask_1, labels) in enumerate(tqdm(test_loader)):
            batch = len(img_batch)
            with torch.set_grad_enabled(True):
                y = model(imgs=img_batch.cuda(), texts=embed_batch1.cuda(), mask_batch=mask_batch1.cuda(), img_edge_index=img_edge_index,
                      t1_token_length=org_token_len, t1_word_seq=org_seq, txt_edge_index=edge_cap1, gnn_mask=gnn_mask_1.cuda(),
                      np_mask=np_mask_1.cuda(), img_edge_attr=None)
                loss = cross_entropy_loss(y, labels.cuda())
                test_loss += float(loss.clone().detach().item())
            predict = predict + get_metrics(y.cpu())
            real_label = real_label + labels.cpu().numpy().tolist()
            torch.cuda.empty_cache()
            del img_batch, embed_batch1
    acc, recall, precision, f1 = get_four_metrics(real_label, predict)

    logger.info('Test Epoch: {} Avg loss: {:.4f} Acc: {:.4f}  Rec: {:.4f} Pre: {:.4f} F1: {:.4f}\n'.format(epoch, test_loss / len(test_loader),
                                                                                                       acc, recall, precision, f1))
    return f1


def test_match_accuracy(val_loader,model,cross_entropy_loss,args,img_edge_index):
    """
    Args:
        Once the model is trained, it is used to evaluate the how accurately the captions align with the objects in the image
    """
    try:
        print("Loading Saved Model")
        checkpoint = torch.load(args.save)
        model.load_state_dict(checkpoint)
        print("Saved Model successfully loaded")
        val_loss = 0.
        predict = []
        real_label = []
        model.eval()
        with torch.no_grad():
            # 在模型
            for batch_idx, (img_batch, embed_batch1, org_seq, org_word_len, org_token_len, mask_batch1,
                            edge_cap1, gnn_mask_1, np_mask_1, labels) in enumerate(tqdm(val_loader)):
                with torch.no_grad():
                    y = model(imgs=img_batch.cuda(), texts=embed_batch1.cuda(), mask_batch=mask_batch1.cuda(), img_edge_index=img_edge_index,
                      t1_token_length=org_token_len, t1_word_seq=org_seq, txt_edge_index=edge_cap1, gnn_mask=gnn_mask_1.cuda(),
                      np_mask=np_mask_1.cuda(), img_edge_attr=None)

                    loss = cross_entropy_loss(y, labels.cuda())
                    val_loss += float(loss.clone().detach().item())
                predict = predict + get_metrics(y.cpu())
                real_label = real_label + labels.cpu().numpy().tolist()
                torch.cuda.empty_cache()
                del img_batch, embed_batch1
            acc, recall, precision, f1 = get_four_metrics(real_label, predict)

        print("Avg loss: {:.4f} Acc: {:.4f}  Rec: {:.4f} Pre: {:.4f} F1: {:.4f}".format(val_loss, acc, recall, precision, f1))
    except Exception as e:
        print(e)
        exit()


def main(nni_params):
    with open('parameter.json') as f:
        parameter = json.load(f)
    seed_everything(nni_params["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--mode', type=str, default='train',
                        help="mode, {'" + "train" + "', '" + "eval" + "'}")
    args = parser.parse_args()

    annotation_files = parameter["annotation_files"]
    img_files = parameter["DATA_DIR"]
    use_np = bool(nni_params["use_np"])


    model = KEHModel(txt_input_dim=parameter["txt_input_dim"], txt_out_size=nni_params["txt_out_size"],
                     rnn_type=parameter["rnn_type"],
                     txt_rnn_layers=nni_params["txt_rnn_layers"], txt_bidirectional=parameter["txt_bidirectional"],
                     txt_rnn_drop=nni_params["txt_rnn_drop"], img_input_dim=parameter["img_input_dim"],
                     img_inter_dim=parameter["img_inter_dim"],
                     img_out_dim=parameter["img_out_dim"], cro_layers=nni_params["cro_layers"],
                     cro_heads=parameter["cro_heads"], cro_drop=nni_params["cro_drop"],
                     txt_gat_layer=nni_params["txt_gat_layer"], txt_gat_drop=nni_params["txt_gat_drop"],
                     txt_gat_head=nni_params["txt_gat_head"],
                     txt_self_loops=bool(nni_params["txt_self_loops"]), img_gat_layer=parameter["img_gat_layer"],
                     img_gat_drop=parameter["img_gat_drop"],
                     img_gat_head=parameter["img_gat_head"], img_self_loops=bool(nni_params["img_self_loops"]),
                     img_edge_dim=parameter["img_edge_dim"],
                     img_patch=parameter["img_patch"])

    model.to(device=device)
    optimizer = optim.Adam(params=model.parameters(), lr=nni_params["lr"], betas=(0.9, 0.999), eps=1e-8,
                           weight_decay=nni_params["weight_decay"], amsgrad=True)
    # optimizer = optim.Adam(params=model.parameters(), lr=parameter["lr"], betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=True)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=nni_params["patience"], verbose=True,
                                  warm_up_epoch=nni_params["warm_up_epoch"],warm_up_decrease=nni_params["warm_up_decrease"])
    cross_entropy_loss = CrossEntropyLoss()

    print("Total Params", sum(p.numel() for p in model.parameters() if p.requires_grad))
    # args.path must be relative path
    img_edge_index = construct_edge_image(parameter["img_patch"])

    if args.mode == 'train':
        annotation_train = os.path.join(annotation_files, "traindep.json")
        annotation_val = os.path.join(annotation_files, "valdep.json")
        annotation_test = os.path.join(annotation_files, "testdep.json")
        img_train = os.path.join(img_files, "train_B32.pt")
        img_val = os.path.join(img_files, "val_B32.pt")
        img_test = os.path.join(img_files, "test_B32.pt")
        train_dataset = BaseSet(type = "train", max_length = parameter["max_length"], text_path = annotation_train, use_np =use_np, img_path = img_train)
        val_dataset = BaseSet(type = "val", max_length = parameter["max_length"], text_path = annotation_val, use_np =use_np, img_path = img_val)
        test_dataset = BaseSet(type = "test", max_length = parameter["max_length"], text_path = annotation_test, use_np =use_np, img_path = img_test)
        train_loader = DataLoader(dataset=train_dataset, batch_size=nni_params["batch_size"], num_workers=4, shuffle=True,
                                  collate_fn=PadCollate(use_np=use_np))
        print("training dataset has been loaded successful!")
        val_loader = DataLoader(dataset=val_dataset, batch_size=nni_params["batch_size"], num_workers=4, shuffle=True,
                                collate_fn=PadCollate(use_np=use_np))
        print("validation dataset has been loaded successful!")
        test_loader = DataLoader(dataset=test_dataset, batch_size=nni_params["batch_size"], num_workers=4, shuffle=True,
                                collate_fn=PadCollate(use_np=use_np))
        print("test dataset has been loaded successful!")
        start_epoch = 0
        patience = 10
        try:
            print("Loading Saved Model")
            checkpoint = torch.load(args.save)
            model.load_state_dict(checkpoint)
            start_epoch = int(re.search("-\d+", args.save).group()[1:]) + 1
            print("Saved Model successfully loaded")
            # Only effect special layers like dropout layer
            model.eval()
            best_loss = eval_validation_loss(val_loader=val_loader)
        except:
            best_loss = np.Inf
        early_stop = False
        counter = 0
        best_f1_test = 0
        for epoch in range(start_epoch+1, parameter["epochs"] + 1):
            # Training epoch
            train_model(epoch=epoch, train_loader=train_loader,model=model,cross_entropy_loss=cross_entropy_loss,optimizer=optimizer,
                        img_edge_index=img_edge_index)
            # Validation epoch
            avg_val_loss = evaluate_model(epoch, val_loader=val_loader,model=model,cross_entropy_loss=cross_entropy_loss,
                                          img_edge_index=img_edge_index)
            f1_test = evaluate_model_test(epoch, test_loader=test_loader,model=model,cross_entropy_loss=cross_entropy_loss,
            img_edge_index=img_edge_index)
            nni.report_intermediate_result(f1_test)
            logger.debug('test f1 score %f', f1_test)
            logger.debug('Pipe send intermediate result done.')
            # final best_f1_test update
            if best_f1_test<f1_test:
                best_f1_test=f1_test
            scheduler.step(avg_val_loss)
            if avg_val_loss <= best_loss:
                counter = 0
                best_loss = avg_val_loss
                print("Best model updated.")
                torch.cuda.empty_cache()
            else:
                counter += 1
                if counter >= patience:
                    early_stop = True
            # If early stopping flag is true, then stop the training
            if early_stop:
                logger.debug("\nEarly stopping\n")
                break
        nni.report_final_result(best_f1_test)
        logger.debug('Final result is %g', best_f1_test)
        logger.debug('Send final result done.')
    elif args.mode == 'eval':
        print("no eval for nni")
        # # args.save 制定路径
        # annotation_test = os.path.join(annotation_files, "testdep.json")
        # img_test = os.path.join(img_files, "test_B32.pt")
        # test_dataset = BaseSet(type="test", max_length=parameter["max_length"], text_path=annotation_test, use_np=use_np,
        #                        img_path=img_test)
        # test_loader = DataLoader(dataset=test_dataset, batch_size=parameter["batch_size"], num_workers=4, shuffle=True,
        #                          collate_fn=PadCollate(use_np=use_np))
        #
        # print("validation dataset has been loaded successful!")
        # test_match_accuracy(val_loader=test_loader)

    else:
        print("Mode of SSGN is error!")


if __name__ == "__main__":

    try:
        params = nni.get_next_parameter()
        logger.debug(params)
        print(params)
        main(nni_params=params)
    except Exception as exception:
        logger.exception(exception)
        raise
