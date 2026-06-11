from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.models.model_factory import load_checkpoint


def build_inference_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def predict_image(
    model: torch.nn.Module,
    image_path: Path,
    class_names: list[str],
    image_size: int,
    device: str = "cpu",
) -> dict[str, object]:
    transform = build_inference_transform(image_size)
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        score, index = torch.max(probabilities, dim=0)

    return {
        "label": class_names[index.item()],
        "confidence": score.item(),
        "probabilities": probabilities.cpu().numpy(),
        "image": image,
        "index": index.item(),
    }


def load_model_for_inference(checkpoint_path: Path, model_name: str, num_classes: int, device: str = "cpu") -> tuple[torch.nn.Module, dict[str, object]]:
    model, metadata = load_checkpoint(str(checkpoint_path), model_name, num_classes=num_classes, device=device)
    return model, metadata
