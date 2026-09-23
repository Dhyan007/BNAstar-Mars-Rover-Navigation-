import os
import random
import cv2
import numpy as np
import pandas as pd
import torch


def get_two_random_images(IMAGES_PATH):
    images = [
        f for f in os.listdir(IMAGES_PATH)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    selected = random.sample(images, 2)

    return [
        os.path.join(IMAGES_PATH, selected[0]),
        os.path.join(IMAGES_PATH, selected[1])
    ]


def get_random_images_and_probabilities(images_path, model, num_images=2):

    image_files = sorted([
        f for f in os.listdir(images_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if len(image_files) < num_images:
        raise ValueError(
            f"Only {len(image_files)} images found, "
            f"but {num_images} were requested."
        )

    # Random unique indices
    random_indices = random.sample(
        range(len(image_files)),
        num_images
    )

    device = next(model.parameters()).device

    results = []

    for image_index in random_indices:

        image_file = image_files[image_index]

        image_path = os.path.join(
            images_path,
            image_file
        )

        image = cv2.imread(image_path)

        if image is None:
            continue

        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image_tensor = (
            torch.from_numpy(image_rgb)
            .float() / 255.0
        )

        image_tensor = image_tensor.permute(2, 0, 1)

        input_tensor = (
            image_tensor
            .unsqueeze(0)
            .to(device)
        )

        with torch.no_grad():

            output = model(input_tensor)

            if isinstance(output, dict):
                output = output["out"]

            probabilities = torch.softmax(
                output,
                dim=1
            )

        probabilities = probabilities.squeeze(0)

        class_probabilities = (
            probabilities
            .mean(dim=(1, 2))
            .cpu()
            .numpy()
        )

        results.append({
            "image_index": image_index,
            "image": image_file,

            "soil_prob": float(class_probabilities[0]),
            "bedrock_prob": float(class_probabilities[1]),
            "sand_prob": float(class_probabilities[2]),
            "rock_prob": float(class_probabilities[3]),
            "unknown": float(class_probabilities[4])
        })

    return results
