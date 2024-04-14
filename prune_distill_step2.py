import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from segment_anything_kd import SamPredictor, sam_model_registry
from segment_anything_kd.modeling.image_encoder import Attention
from load_sam_json import SamDataset
from torch.nn.functional import threshold, normalize
from segment_anything_kd.utils.transforms import ResizeLongestSide
from prune_funcs import calculate_iou, get_pos_init, del_pos_init, prune_sam_step1 ,prune_sam_step2_global
import torch_pruning as tp
import copy
import json
from pycocotools import mask as mask_utils
import argparse
import os
join = os.path.join
import glob
import monai


seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser(description='SlimSAM')
parser.add_argument('--traindata_path', type=str,default = '')
parser.add_argument('--valdata_path', type=str,default = '')
parser.add_argument('--trainsize', type=int,default = 10000)
parser.add_argument('--gradsize', type=int,default = 1000)
parser.add_argument('--valsize', type=int,default = 50)
parser.add_argument('--epochs', type=int,default = 20)
parser.add_argument('--norm_type', type=str,default = 'gaussian')
parser.add_argument('--imptype', type=str,default = 'Disturb')
parser.add_argument('--global_way', type=bool,default = False)
parser.add_argument('--prune_ratio', type=float,default = 0.5)
parser.add_argument('--model_path', type=str,default = 'checkpoints/vit_b_medslim_step1_.pth')
args, unparsed = parser.parse_known_args()           

class NpyDataset(Dataset):
    def __init__(self, data_root, dataset_size, bbox_shift=20):
        self.data_root = data_root
        self.dataset_size = dataset_size
        self.gt_path = join(data_root, "gts")
        self.img_path = join(data_root, "imgs")
        self.gt_path_files = sorted(
            glob.glob(join(self.gt_path, "**/*.npy"), recursive=True)
        )
        self.gt_path_files = [
            file
            for file in self.gt_path_files
            if os.path.isfile(join(self.img_path, os.path.basename(file)))
        ]
        self.bbox_shift = bbox_shift
        print(f"number of images: {len(self.gt_path_files)}")

    def __len__(self):
        # return len(self.gt_path_files)
        return self.dataset_size

    def __getitem__(self, index):
        # load npy image (1024, 1024, 3), [0,1]
        img_name = os.path.basename(self.gt_path_files[index])
        img_1024 = np.load(
            join(self.img_path, img_name), "r", allow_pickle=True
        )  # (1024, 1024, 3)
        # convert the shape to (3, H, W)
        img_1024 = np.transpose(img_1024, (2, 0, 1))
        assert (
            np.max(img_1024) <= 1.0 and np.min(img_1024) >= 0.0
        ), "image should be normalized to [0, 1]"
        gt = np.load(
            self.gt_path_files[index], "r", allow_pickle=True
        )  # multiple labels [0, 1,4,5...], (256,256)
        assert img_name == os.path.basename(self.gt_path_files[index]), (
            "img gt name error" + self.gt_path_files[index] + self.npy_files[index]
        )
        label_ids = np.unique(gt)[1:]
        gt2D = np.uint8(
            gt == random.choice(label_ids.tolist())
        )  # only one label, (256, 256)
        assert np.max(gt2D) == 1 and np.min(gt2D) == 0.0, "ground truth should be 0, 1"
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = gt2D.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        bboxes = np.array([x_min, y_min, x_max, y_max])
        return {
            "input_image": torch.tensor(img_1024).float(),
            "gt": torch.tensor(gt2D[None, :, :]).long(),
            "bboxes": torch.tensor(bboxes).float(),
            "names_temp": img_name
        }
    
class MedSAM(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, box):
        image_embedding,_ = self.image_encoder(image)  # (B, 256, 64, 64)
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]  # (B, 1, 4)

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks

def train_model():

    # torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    print("CUDA visible devices: " + str(torch.cuda.device_count()))
    print("CUDA Device Name: " + str(torch.cuda.get_device_name(device)))

    train_root_folder = args.traindata_path
    val_root_folder = args.valdata_path
    TRAIN_SIZE = args.trainsize
    VAL_SIZE = args.valsize
    GRAD_SIZE = args.gradsize
    num_train_epochs = args.epochs
    batch_size = 1
    model_path = args.model_path


    # Creating dataset loaders
    grad_dataset = NpyDataset(data_root=train_root_folder, dataset_size=GRAD_SIZE)
    grad_loader = DataLoader(dataset=grad_dataset, batch_size=1, shuffle=False, num_workers=4,
                              pin_memory=True, drop_last=True)

    train_dataset = NpyDataset(data_root=train_root_folder, dataset_size=TRAIN_SIZE)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                              pin_memory=True, drop_last=True)

    val_dataset = NpyDataset(data_root=val_root_folder, dataset_size=VAL_SIZE)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=2,
                             pin_memory=True, drop_last=False)

    # student model
    model = torch.load(model_path)
    model.image_encoder = model.image_encoder.module

    # rewrite the forward function of image encoder
    def forward(self, x):

        block_outputs = []
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
            
        block_outputs.append(x)

        for blk in self.blocks:
            x,qkv_emb,mid_emb,x_emb = blk(x)
            block_outputs.append(x_emb)

        x = self.neck(x.permute(0, 3, 1, 2))
        
        block_emb = block_outputs[0]
        for emb in block_outputs[1:]:
            block_emb = torch.cat([block_emb,emb],dim=0)
        
        return x, block_emb


    # teacher model
    teacher_model_type = 'vit_b'
    checkpoint = 'checkpoints/medsam_vit_b_qkv.pth'
    teacher_model = sam_model_registry[teacher_model_type](checkpoint=checkpoint)
    teacher_model.to(device)
    teacher_model.eval()

    # load the pruned model
    pruned_model = torch.load(model_path)
    pruned_model.image_encoder = pruned_model.image_encoder.module
    pruned_model.to(device)
    pruned_model.eval()

    # Rewrite forward functions
    import types
    funcType = types.MethodType
    model.image_encoder.forward = funcType(forward, model.image_encoder)
    pruned_model.image_encoder.forward = funcType(forward, pruned_model.image_encoder)
    teacher_model.image_encoder.forward = funcType(forward, teacher_model.image_encoder)
    


    MSE_loss = torch.nn.MSELoss()
    lr = 1e-4
    ratio = args.prune_ratio
    loss_fn = torch.nn.MSELoss()
    transform = ResizeLongestSide(1024)
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    norm_type = args.norm_type
    imptype = args.imptype
    global_way = args.global_way
    a_weight = 0.5
    round_to = model.image_encoder.num_heads

    print("===========================Parameter Settings===========================")

    print("Pruning Ratio:",ratio)
    print("VIT num_heads:",round_to)
    print("norm_type:",norm_type)
    print("imptype:",imptype)
    print("global:",global_way)
    print("learning rate:",lr)
    print("a_weight:",a_weight)
    print("round_to",round_to)
    print("TRAIN_SIZE",TRAIN_SIZE,"VAL_SIZE",VAL_SIZE, "GRAD_SIZE",GRAD_SIZE,"Epochs",num_train_epochs)

    model_name = teacher_model_type
    example_inputs = torch.randn(1, 3, 1024, 1024)

    for k in range(1):
############################################get initial grad for importance estimation############################################
        best_loss = np.inf
        model.to(device)
        model.image_encoder.train()
        grad_iter = iter(grad_loader)

        for i in range(len(grad_iter)):

            batch = next(grad_iter)
            input_image = batch["input_image"].to(device)

            with torch.no_grad():
                teacher_embedding,_ = pruned_model.image_encoder(input_image)
                teacher_embedding += torch.normal(mean=0,std=0.01,size=(1, 256, 64, 64)).to(device)   #Disturbed image embedding
                
            student_embedding, _= model.image_encoder(input_image)
            
            loss = loss_fn(teacher_embedding, student_embedding)
            loss.backward()

    
        #########################################################################################################
        print("===========================Pruning Start===========================")
        #Bottleneck Pruning
        model.cpu().eval()
        model = del_pos_init(model)
        ##Global pruning QKV Attention
        model.image_encoder = prune_sam_step2_global(model=model.image_encoder, example_inputs=example_inputs, model_name=model_name, round_to=round_to, ratio=ratio, imptype = imptype, norm_type=norm_type, global_way=global_way, gs=1)
        ##Global pruning MLP Layer
        model.image_encoder = prune_sam_step2_global(model=model.image_encoder, example_inputs=example_inputs, model_name=model_name, round_to=round_to, ratio=ratio, imptype = imptype, norm_type=norm_type, global_way=global_way, gs=2)

        model = get_pos_init(model)
        model.to(device)

        model.image_encoder = torch.nn.DataParallel(model.image_encoder)
        model.image_encoder.train()
        optimizer = torch.optim.Adam(model.image_encoder.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',factor=0.5,patience=4,verbose=True)

        model.zero_grad()
        teacher_model.zero_grad()

        #Embedding Aligning
        for epoch in range(num_train_epochs):

            torch.cuda.empty_cache()
            train_iter = iter(train_loader)

            if epoch<11:
                a_weight = (11-epoch-1)/11
                print("Dynamic weight:",a_weight)
            else:
                a_weight = 0
                print("Dynamic weight:",a_weight)

            for i in range(len(train_iter)):

                batch = next(train_iter)
                input_image = batch["input_image"].to(device)

                with torch.no_grad():
                    teacher_embedding,teacher_block_emb = teacher_model.image_encoder(input_image)
                    pruned_embedding,pruned_block_emb = pruned_model.image_encoder(input_image)

                student_embedding,student_block_emb = model.image_encoder(input_image)

                #loss = loss_fn(student_embedding, teacher_embedding)
                loss = (1-a_weight)*loss_fn(student_embedding, teacher_embedding)+a_weight*loss_fn(student_block_emb, pruned_block_emb)+a_weight*loss_fn(student_embedding, pruned_embedding)
                loss.backward()
                    
                #### batchsize×4 ####
                if i%4==3:
                    optimizer.step()
                    optimizer.zero_grad()
                
                #validation
                medsam_model = MedSAM(
                    image_encoder=model.image_encoder,
                    mask_decoder=model.mask_decoder,
                    prompt_encoder=model.prompt_encoder,
                ).to(device)                    

                if i == len(train_iter)-1:
                    total_loss = 0
                    total_samples = 0
                    model.image_encoder.eval()
                    with torch.no_grad():
                        val_iter = iter(val_loader)
                        for j in range(len(val_iter)):
                            batch = next(val_iter)
                            sub_count = batch["input_image"].size(0)

                            image = batch["input_image"].to(device)
                            gt2D = batch["gt"].to(device)
                            boxes = batch["bboxes"].to(device)

                            boxes_np = boxes.detach().cpu().numpy()
                            image, gt2D = image.to(device), gt2D.to(device)
                            medsam_pred = medsam_model(image, boxes_np)
                            sub_loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float())
                                
                            total_loss += sub_loss
                            total_samples += sub_count

                    total_loss = total_loss/total_samples
                    model.image_encoder.train()

                    model.image_encoder.eval()
                    if total_loss<=best_loss:
                        best_loss = total_loss
                        if global_way:
                            prune_type = "global"
                        else:
                            prune_type = "local"
                        filename = f"checkpoints/vit_b_medslim_final_step2_{ratio}_{prune_type}_{norm_type}.pth"
                        torch.save(model, filename)   
                        print("save checkpoint")
                    model.image_encoder.train()

                    scheduler.step(total_loss)

                    print("epoch:",epoch)
                    print("Loss: {} Best Loss {}".format(total_loss,best_loss))



        




        


        


    






           


            











if __name__ == '__main__':
    train_model()
