from tqdm import tqdm
import torch
import os
import wandb
from sys import argv
import math
import numpy as np
from torchvision.ops import sigmoid_focal_loss
from torch.optim.lr_scheduler import _LRScheduler
from architecture.model_gnn import Model
from architecture.model_set import SetModel
from architecture.config import config_runtime_set as config_runtime
from architecture.config import config_model, config_data
from utils_set import setup_dataloader, collate_set_transformer
from scoring import bc_scoring, bc_score_names, nanmean

model1_num = argv[1]
model_num = argv[2]

wandb.login(key='aa3d83b08d1587884348defb38d3143f893e2b96')

class WarmUpCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, T_max, T_warmup, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.T_warmup = T_warmup
        self.eta_min = eta_min
        super(WarmUpCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.T_warmup:
            return [base_lr * self.last_epoch / self.T_warmup for base_lr in self.base_lrs]
        else:
            k = 1 + math.cos(math.pi * (self.last_epoch - self.T_warmup) / (self.T_max - self.T_warmup))
            return [self.eta_min + (base_lr - self.eta_min) * k / 2 for base_lr in self.base_lrs]

def scoring(eval_results, device=torch.device('cpu')):
    # compute sum losses and scores for each entry
    losses, scores = [], []
    for loss, y, p in eval_results:
        losses.append(loss)
        scores.append(bc_scoring(y, p))

    # average scores
    m_losses = np.mean(losses)
    m_scores = nanmean(torch.stack(scores, dim=0)).numpy()

    # pack scores
    scores = {'loss': float(m_losses)}
    for i,s in enumerate(m_scores.squeeze(1)):
        scores[f'{bc_score_names[i]}'] = s
    return scores
    
def eval_step(model1, model2, device, batch_data, config_runtime, global_step):
    model1.eval()
    # get embs
    seq, SCOV, nn_ids, D, R, SCOD, y  = [data for data in batch_data]
    emb_list=[]
    for i in range(len(SCOV)):
        num_nodes, k = nn_ids[i].shape
        edge_src = torch.arange(num_nodes).unsqueeze(1).repeat(1, k).flatten()
        edge_dst = nn_ids[i].flatten()
        with torch.no_grad():
            emb = model1([[seq.to(device), SCOV[i].to(device)], [SCOD[i].to(device), R[i].to(device), D[i].to(device)]],edge_src.to(device),edge_dst.to(device), get_mor=True)
        emb_list.append(emb)
    emb_list = torch.stack(emb_list, dim=1)
    # evalluate with setmodeli
    z = model2.forward(emb_list)
    # compute weighted loss
    loss = sigmoid_focal_loss(z, y.to(z.dtype).to(device), alpha=config_runtime['loss_alpha'], gamma=config_runtime['loss_gamma'], reduction='mean')
    s = torch.sigmoid(z)
    return loss, y.detach(), torch.sigmoid(z).detach()

def train(config_data, config_model, config_runtime, output_path):
    wandb.init(project="PL-BS", name=f"run{model_num}")
    
    # define device
    device = torch.device(config_runtime['device'])

    # Load model
    model1 = Model(config_model).to(device)
    ckpt = torch.load(f"model_{model1_num}.pt", map_location=device, weights_only=True)
    model1.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    
    # create model
    model = SetModel(config_model).to(device)
    #print(model)
    print(f"> {sum([int(torch.prod(torch.tensor(p.shape))) for p in model.parameters()])} parameters")
    
    # if the model should be continued
    model_filepath = 'model_ckpt_XX.pt'
    if os.path.isfile(model_filepath):
        checkpoint = torch.load(model_filepath, map_location=lambda storage, loc: storage.cuda(0))
        model.load_state_dict(checkpoint)
        #model.load_state_dict(torch.load(model_filepath))
        global_step = 5886
    else:
        # starting global step
        global_step = 0
    
   
    # min loss initial value
    max_PR = 1e-9
    patience_counter = 0
    
    # setup dataloaders - feature orders: onehot_seq, rmsf1, rmsf2, rsa, angular_variation, nn_topk, D_nn, R_nn, SCOD_nn, motion_v, motion_s, y
    dataloader_train = setup_dataloader(config_data, config_data['train_selection_filepath'])
    dataloader_test = setup_dataloader(config_data, config_data['valid_selection_filepath'])
    val_iterator = iter(dataloader_test)
    
    # Optimiser
    steps_per_epoch = len(dataloader_train)
    total_steps = config_runtime['num_epochs'] * steps_per_epoch
    warmup_steps = config_runtime['warmup_epochs'] * config_runtime['log_step']
    optimizer = torch.optim.AdamW(model.parameters(), lr=config_runtime['max_lr'])
    scheduler = WarmUpCosineAnnealingLR(optimizer, T_max=total_steps, T_warmup=warmup_steps, eta_min=config_runtime['min_lr'])

    # quick training step on largest data: memory check and pre-allocation
    batch_data = collate_set_transformer([dataloader_train.dataset.get_largest()])
    optimizer.zero_grad()
    loss, _, _ = eval_step(model1, model, device, batch_data, config_runtime, global_step)
    loss.backward()
    optimizer.step()

    # start training
    for epoch in range(config_runtime['num_epochs']):
        print (f"Epoch {epoch}.............Current LR: {scheduler.get_last_lr()[0]:.8f}")
        # train mode
        model = model.train()

        # train model
        train_results = []
        for batch_train_data in tqdm(dataloader_train):
            # global step
            global_step += 1
            
            # set gradient to zero
            optimizer.zero_grad()

            # forward & backward propagation
            loss, y, p = eval_step(model1, model, device, batch_train_data, config_runtime, global_step)
            loss.backward()
            optimizer.step()
            scheduler.step()
            # store evaluation results
            train_results.append([loss.detach().cpu(), y.cpu(), p.cpu()])
            
            if (global_step+1) % config_runtime["log_step"] == 1:
                # process evaluation results
                with torch.no_grad():
                    # scores evaluation results and reset buffer
                    scores = scoring(train_results, device=device)
                    scores_ = {f"{k}/train": v for k, v in scores.items()}
                    wandb.log(scores_, step=(global_step))
                    train_results = []

                    # save model checkpoint
                    model_filepath = os.path.join(output_path, f'model_ckpt_{model_num}.pt')
                    torch.save(model.state_dict(), model_filepath)
                    
            # evaluation step
            if (global_step+1) % config_runtime["log_step"] == 0:
                # evaluation mode
                model = model.eval()

                with torch.no_grad():
                    test_results = []
                    batches_to_eval = config_runtime["log_step"]

                    # Loop for the specified number of validation batches
                    for _ in range(batches_to_eval):
                        try:
                            # Get the next batch from the persistent iterator
                            batch_test_data = next(val_iterator)
                        except StopIteration:
                            # If the iterator is exhausted, reset it and get the first batch
                            #print("\nValidation iterator reset.")
                            val_iterator = iter(dataloader_test)
                            batch_test_data = next(val_iterator)

                        # Perform the forward pass and calculate loss
                        losses, y, p = eval_step(model1, model, device, batch_test_data, config_runtime, global_step)

                        # Store evaluation results
                        test_results.append([losses.detach().cpu(), y.cpu(), p.cpu()])

                    # scores evaluation results
                    scores = scoring(test_results, device=device)
                    scores_ = {f"{k}/valid": v for k, v in scores.items()}
                    wandb.log(scores_, step=(global_step))

                    # save model and update min loss
                    current_PR = scores['PR']
                    if max_PR <= current_PR:
                        # update min loss
                        max_PR = current_PR
                        # save model
                        model_filepath = os.path.join(output_path, f'model_{model_num}.pt')
                        torch.save(model.state_dict(), model_filepath)
                        # Early stopping check
                        patience_counter = 0
                    else:
                        patience_counter += 1

                # back in train mode
                model = model.train()
        if patience_counter >= config_runtime['patience']:
            break  # Break out of the training loop
    wandb.finish()


train(config_data, config_model, config_runtime, '.')
