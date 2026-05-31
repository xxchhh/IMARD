import argparse
import csv
import glob
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from model import create_model


IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")


def build_normalizer(mean, std):
    mean_array = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std_array = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    return mean_array, std_array


def read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def preprocess_image(path, image_size, mean, std):
    image = read_image(path)
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32) / 255.0
    mean_array, std_array = build_normalizer(mean, std)
    image = (image - mean_array) / std_array
    image = image.transpose(2, 0, 1)
    tensor = torch.from_numpy(image).float().unsqueeze(0)
    return tensor


def load_state_dict(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        cleaned_key = key[7:] if key.startswith("module.") else key
        cleaned_state_dict[cleaned_key] = value
    return cleaned_state_dict


def collect_images(input_path):
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]
    image_paths = []
    for pattern in IMAGE_EXTENSIONS:
        image_paths.extend(input_path.rglob(pattern))
    return sorted(image_paths)


def predict_image(model, image_path, device, image_size, mean, std, num_digits):
    tensor = preprocess_image(image_path, image_size, mean, std).to(device)
    with torch.no_grad():
        logits, aux_info = model(tensor)
        probabilities = torch.softmax(logits[0], dim=1)
        confidence_values, digits = torch.max(probabilities, dim=1)
        prediction = "".join(str(int(digit)) for digit in digits[:num_digits].cpu())
        confidence = float(confidence_values[:num_digits].mean().cpu())
        quality = aux_info.get("quality_info", {})
        quality_output = {}
        for key, value in quality.items():
            quality_output[key] = float(value[0].detach().cpu())
    return prediction, confidence, quality_output


def save_results(results, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["image", "prediction", "confidence", "clarity", "occlusion", "illumination"])
        writer.writeheader()
        for result in results:
            writer.writerow(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-digits", type=int, default=5)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--backbone-channels", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = create_model(
        num_digits=args.num_digits,
        num_classes=args.num_classes,
        device=device,
        backbone_channels=args.backbone_channels
    )
    state_dict = load_state_dict(args.checkpoint, device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image_paths = collect_images(args.input)
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {args.input}")

    results = []
    for image_path in image_paths:
        prediction, confidence, quality = predict_image(
            model=model,
            image_path=image_path,
            device=device,
            image_size=args.image_size,
            mean=mean,
            std=std,
            num_digits=args.num_digits
        )
        result = {
            "image": str(image_path),
            "prediction": prediction,
            "confidence": confidence,
            "clarity": quality.get("clarity", ""),
            "occlusion": quality.get("occlusion", ""),
            "illumination": quality.get("illumination", "")
        }
        results.append(result)
        print(f"{image_path}: {prediction} ({confidence:.4f})")

    save_results(results, args.output)
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
