import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import VOCDataset
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
)
from loss import YoloLoss
import matplotlib.pyplot as plt
import time

seed = 123
torch.manual_seed(seed)

# Hyperparameters
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16 # 64 in original paper
WEIGHT_DECAY = 0
EPOCHS = 100
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "overfit.pth.tar"
IMG_DIR = "pascalVOC/images"
LABEL_DIR = "pascalVOC/labels"

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes

transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])

def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")
    return mean_loss

def main():
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss()

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    train_dataset = VOCDataset(
        "pascalVOC/100examples.csv",
        # "pascalVOC/8examples.csv",
        # "pascalVOC/train.csv",
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )

    test_dataset = VOCDataset(
        "pascalVOC/test.csv", 
        transform=transform, 
        img_dir=IMG_DIR, 
        label_dir=LABEL_DIR,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
        # drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )
    
    losses = []
    mAPs = []
    for epoch in range(EPOCHS):
        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print(f"Train mAP: {mean_avg_prec}")
        mAPs.append(mean_avg_prec)

        mean_loss = train_fn(train_loader, model, optimizer, loss_fn)
        losses.append(sum(mean_loss) / len(mean_loss))

        # for x, y in train_loader:
        #    x = x.to(DEVICE)
        #    for idx in range(8):
        #        bboxes = cellboxes_to_boxes(model(x))
        #        bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
        #        plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes)

        #    import sys
        #    sys.exit()

        #if mean_avg_prec > 0.9:
        #    checkpoint = {
        #        "state_dict": model.state_dict(),
        #        "optimizer": optimizer.state_dict(),
        #    }
        #    save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
        #    import time
        #    time.sleep(10)

    # Save losses and mAPs to files
    with open("100_img_standard/lists/losses.txt", "w") as f:
        for loss in losses:
            f.write(f"{loss}\n")

    with open("100_img_standard/lists/mAPs.txt", "w") as f:
        for mAP in mAPs:
            f.write(f"{mAP}\n")

    # Plot mean losses
    plt.figure()
    plt.plot(losses, label="Mean Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Loss")
    plt.title("Mean Loss per Epoch")
    plt.legend()
    plt.savefig("100_img_standard/plots/100_img_loss.png")
    plt.show()

    # Plot mAP
    plt.figure()
    plt.plot(mAPs, label="mAP")
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.title("mAP per Epoch")
    plt.legend()
    plt.savefig("100_img_standard/plots/100_img_mAP.png")
    plt.show()



if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    with open("100_img_standard/lists/time.txt", "w") as f:
        f.write(f"Start Time: {start_time:.2f} seconds\n")
        f.write(f"End Time: {end_time:.2f} seconds\n")
        f.write(f"Time taken: {end_time - start_time:.2f} seconds\n")