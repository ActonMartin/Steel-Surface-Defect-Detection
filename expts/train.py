import model
import dataset
import wandb
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import lovasz_softmax
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import random
import cv2
import warnings
warnings.filterwarnings("ignore")

class SegmentationMetrics:
    def __init__(self, n_classes, device, ignore=None):
        self.n_classes = n_classes
        self.device = device
        self.ignore = torch.tensor(ignore).long()
        self.include = torch.tensor(
            [n for n in range(self.n_classes) if n not in self.ignore]
        ).long()

    def reset(self):
        self.confusion_matrix = torch.zeros(
            (self.n_classes, self.n_classes), device=self.device
        ).long()
        self.ones = None
        self.last_scan_size = None

    def addbatch(self, preds, targets):
        preds_row = preds.reshape(-1)
        targets_row = targets.reshape(-1)
        indices = torch.stack([preds_row, targets_row], dim=0)
        if self.ones is None or self.last_scan_size != indices.shape[-1]:
            self.ones = torch.ones((indices.shape[-1]), device=self.device).long()
            self.last_scan_size = indices.shape[-1]
        self.confusion_matrix = self.confusion_matrix.index_put_(
            tuple(indices), self.ones, accumulate=True
        )

    def getstats(self):
        confusion_matrix = self.confusion_matrix.clone()
        confusion_matrix[self.ignore] = 0
        confusion_matrix[:, self.ignore] = 0
        true_pos = confusion_matrix.diag()
        false_pos = confusion_matrix.sum(dim=1) - true_pos
        false_neg = confusion_matrix.sum(dim=0) - true_pos
        return true_pos, false_pos, false_neg

    def getiou(self):
        true_pos, false_pos, false_neg = self.getstats()
        intersection = true_pos
        union = true_pos + false_pos + false_neg + 1e-15
        iou = intersection / union
        iou_mean = (intersection[self.include] / union[self.include]).mean()
        return iou_mean, iou

    def getdice(self):
        true_pos, false_pos, false_neg = self.getstats()
        numerator = 2*true_pos
        denominator = 2*true_pos + false_neg + false_pos
        dice_mean = numerator[self.include]/denominator[self.include]
        return dice_mean.mean()

    def getacc(self):
        true_pos, false_pos, false_neg = self.getstats()
        total_truepos = true_pos.sum()
        total = true_pos[self.include].sum() + false_pos[self.include].sum() + 1e-15
        accuracy_mean = total_truepos / total
        return accuracy_mean

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.Model(5).to(device)

train_dataset = dataset.SeverstalSteel("severstal-steel-defect-detection", "train", 64, 512)
val_dataset = dataset.SeverstalSteel("severstal-steel-defect-detection", "val", 64, 512)
train_dataloader = DataLoader(train_dataset, 16, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, 16, shuffle=False, num_workers=4)

nll_loss = nn.NLLLoss()
lovaszsoftmax_loss = lovasz_softmax.Lovasz_softmax(ignore=0)
optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0001,
)
evaluator = SegmentationMetrics(5, device, 0)
best = 0
train_step = 0
wandb.init("severstal-steel")
for epoch in range(300):
    loss_cntr = []
    pbar = tqdm(total=len(train_dataloader))
    evaluator.reset()
    for (img, mask) in train_dataloader:
        img, mask = img.to(device), mask.to(device)
        out = model(img)
        loss = nll_loss(torch.log(out.clamp(min=1e-8)), mask.long()) + lovaszsoftmax_loss(out, mask.long())
        if torch.isnan(loss.mean()):
            continue
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        pred_label = out.argmax(dim=1)
        evaluator.addbatch(pred_label, mask.long())
        
        loss_cntr.append(loss.item())
        
        wandb.log({"loss": loss.item(), "train_step": train_step})
        train_step += 1
        pbar.update(1)
    acc = evaluator.getacc()
    iou, c_iou = evaluator.getiou()
    dice_coeff = evaluator.getdice()
    wandb.log({"train_acc": acc.item(), "train_iou": iou.item(), "train_dice": dice_coeff.item(), "train_epoch": epoch})
    pbar.close()
    print(f"\ntrain epoch: {epoch} loss: {round(np.mean(loss_cntr),3)}, , dice coeff: {round(dice_coeff.item(),3)}, iou: {round(iou.item(),3)}, class iou: {[round(c,3) for c in c_iou.detach().cpu().numpy()[1:]]}\n")
    
    if epoch % 10 == 0:
        n_vis = 0
        pbar = tqdm(total=len(val_dataloader))
        evaluator.reset()
        for (img, mask) in val_dataloader:
            img, mask = img.to(device), mask.to(device)
            with torch.no_grad():
                out = model(img)
            pred_label = out.argmax(dim=1)
            evaluator.addbatch(pred_label, mask.long())
            if n_vis < 5 and random.random() < 0.1:
                ind = random.randint(0, img.shape[0]-1)
                im = img[ind].detach().cpu().numpy().transpose(1,2,0)
                pred = pred_label[ind].detach().cpu().numpy().astype(np.float32)
                pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
                m = mask[ind].detach().cpu().numpy().astype(np.float32)
                m = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
                wandb.log({f"{n_vis}": [wandb.Image(np.hstack((m,im,pred)), caption=f"{epoch}")]})
                n_vis += 1
            pbar.update(1)

        acc = evaluator.getacc()
        iou, c_iou = evaluator.getiou()
        dice_coeff = evaluator.getdice()
        wandb.log({"val_acc": acc.item(), "val_iou": iou.item(), "val_dice": dice_coeff.item(), "val_epoch": epoch})
        pbar.close()
        print(f"\nval epoch: {epoch}, dice coeff: {round(dice_coeff.item(),3)}, iou: {round(iou.item(),3)}, class iou: {[round(c,3) for c in c_iou.detach().cpu().numpy()[1:]]}\n")
        if iou.item() > best:
            torch.save(model.state_dict(), "best.ckpt")
            best = iou.item()
            
    torch.save(model.state_dict(), "last.ckpt")
    