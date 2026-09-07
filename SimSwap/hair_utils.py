import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from parsing_model.model import BiSeNet


HAIR_CLASS = 17


def load_hair_model():
    model = BiSeNet(n_classes=19)

    checkpoint = torch.load(
        "./parsing_model/checkpoint/79999_iter.pth",
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(checkpoint)
    model.to("cpu")
    model.eval()

    return model


def get_hair_mask(image, model):
    """
    image: OpenCV BGR image

    returns:
        hair_mask: float mask with shape (H, W), values 0-1
    """

    height, width = image.shape[:2]

    # BiSeNet expects RGB
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pil_image = Image.fromarray(rgb)

    transform = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    tensor = transform(pil_image).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)[0]

    parsing = output.squeeze(0).cpu().numpy()
    parsing = np.argmax(parsing, axis=0)

    # Hair pixels are class 17
    hair_mask = (parsing == HAIR_CLASS).astype(np.uint8)

    # Resize mask back to original image size
    hair_mask = cv2.resize(
        hair_mask,
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )

    return hair_mask