import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
from torchmetrics import Accuracy, ConfusionMatrix, JaccardIndex

from .config import IMAGES_PATH, MASK_PATH_TEST, MODEL_PATH, device


class ImageSegmentationModel(pl.LightningModule):
    def __init__(self, num_classes: int = 5, learning_rate: float = 1e-4):
        super().__init__()

        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.num_classes = num_classes

        self.model_weights = torchvision.models.segmentation.FCN_ResNet50_Weights.DEFAULT
        self.model = torchvision.models.segmentation.fcn_resnet50(weights=self.model_weights)
        self.model.classifier[-1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))

        self.loss = nn.CrossEntropyLoss()
        self.confusion_matrix = ConfusionMatrix(task='multiclass', num_classes=num_classes)
        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.iou = JaccardIndex(task='multiclass', num_classes=num_classes)

    def forward(self, x):
        return self.model(x)['out']

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def load_model(path, device):
    try:
        obj = torch.load(
            path,
            map_location=torch.device("cpu"),
            weights_only=False
        )
    except TypeError:
        obj = torch.load(
            path,
            map_location=torch.device("cpu")
        )

    if isinstance(obj, nn.Module):
        loaded_model = obj
    else:
        raise ValueError(f"Unsupported model format: {type(obj)}")

    loaded_model = loaded_model.to(device)
    loaded_model.eval()

    return loaded_model


def preprocess_image(image_path: str, return_tensor: bool = False):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image_normalized = np.asarray(image, dtype=np.float32) / 255.0

    if return_tensor:
        image_normalized = np.transpose(image_normalized, (2, 0, 1))
        image_tensor = torch.from_numpy(image_normalized).unsqueeze(0)
        return image, image_tensor

    return image


def preprocess_mask(mask_path: str):
    mask = cv2.imread(mask_path, 0)
    mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)
    mask[mask == 255] = 4
    return mask


def display_segmentation(index):
    mask_files = sorted(
        f for f in os.listdir(MASK_PATH_TEST) if f.endswith("_merged.png")
    )
    test_image_name = mask_files[index][:-11] + ".JPG"

    test_image_path = os.path.join(IMAGES_PATH, test_image_name)
    test_image, test_image_tensor = preprocess_image(test_image_path, return_tensor=True)
    test_image_tensor = test_image_tensor.to(device)

    with torch.no_grad():
        prediction = model(test_image_tensor)
        if isinstance(prediction, dict):
            prediction = prediction["out"]
        predicted_mask = torch.argmax(prediction, dim=1).squeeze().cpu().numpy()

    ground_truth_mask_path = os.path.join(MASK_PATH_TEST, test_image_name[:-4] + "_merged.png")
    ground_truth_mask = preprocess_mask(ground_truth_mask_path)

    return test_image, ground_truth_mask, predicted_mask


def initialize_model():
    device_local = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    model = ImageSegmentationModel(num_classes=5)

    state_dict = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state_dict)

    model = model.to(device_local)
    model.eval()

    print("Model loaded successfully!")
    print("Device:", next(model.parameters()).device)

    return model
