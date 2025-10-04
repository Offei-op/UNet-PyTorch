import os
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import json
import numpy as np
import cv2


#Create dataset

def extract_image_path(annotation_file):
    ID = annotation_file['data']['image'][-12:-4]
    return f"survey_plans/anonymised_{ID}.jpg"

def extract_polygon(annotation_file):
    return annotation_file['annotations'][0]['result'][0]['value']['points']

def extract_dimesions(annotation_file):
    return annotation_file['annotations'][0]['result'][0]['original_width'],annotation_file['annotations'][0]['result'][0]['original_height']

def create_img_mask(annotation_file):
    img_path = extract_image_path(annotation_file)
    w,h = extract_dimesions(annotation_file)
    mask = np.zeros((h, w), dtype=np.uint8)

    pts = np.array(extract_polygon(annotation_file), dtype=np.float32)

        # Convert from percentage to pixel coordinates
    pts[:, 0] = pts[:, 0] / 100 * w  # x
    pts[:, 1] = pts[:, 1] / 100 * h  # y
    pts = pts.astype(np.int32)

    # Fill polygon
    cv2.fillPoly(mask, [pts], 1)

    return img_path,mask


class Survey_Plans_Dataset(Dataset):
    def __init__(self, root_path, test=False):

        with open("project-3-at-2025-09-30-22-16-044bef1c.json", "r") as f:
            self.data = json.load(f)

        for annotation_file in self.data:
            self.image_paths,self.masks = create_img_mask(annotation_file)
        
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        mask_array = self.masks[index]
        if mask_array.dtype != np.uint8:
            mask_array = (mask_array * 255).astype(np.uint8)

        mask = Image.fromarray(mask_array).convert("L")

        return self.transform(img), self.transform(mask)
    
    def __len__(self):
        return len(self.images)