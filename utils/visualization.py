from torch import Tensor
from torchvision import transforms
from typing import Iterable
import matplotlib.pyplot as plt

def draw_mask(images, masks, threshold=0.5, alpha=0.6):
    if alpha > 1 or alpha < 0:
        raise ValueError(f"Alpha should be between 0 to 1, but got {alpha}")
    out = images.clone()
    for image, mask in zip(out, masks):
        indice = (mask < threshold).squeeze()
        image[:, indice] *= (1-alpha)
    return out

def show(image:Tensor, title:str=None, dpi:float=100):
    show_image_list(
        images=[image],
        titles=[title],
        shape=(1, 1),
        dpi=dpi
    )

def show_pair(image_pair:Iterable[Tensor], titles:Iterable[str]=None):
    show_image_list(
        images=image_pair,
        titles=titles,
        shape=(1, 2)
    )

def show_masked(images, masks, shape, titles=None):
    show_image_list(
        images=draw_mask(images, masks),
        titles=titles,
        shape=shape,
    )

def show_image_list(
        images:Iterable[Tensor],
        titles:Iterable[str] = None,
        shape:tuple[int, int] = None,
        dpi:float = 300
    ):

    if shape is None:
        shape = (1, len(images))
        
    fig, axarr = plt.subplots(*shape)
    if len(images) > 1:
        axarr = axarr.flatten()
    else:
        axarr = [axarr]
    for image, ax in zip(images, axarr):
        image = image.detach().cpu()
        image = transforms.ToPILImage()(image)
        ax.axis('off')
        if image.mode == 'L':
            ax.imshow(image, cmap='gray')
        else:
            ax.imshow(image)
    if titles:
        for i, ax in enumerate(axarr):
            ax.set_title(titles[i])
    fig.set_dpi(dpi)
    fig.tight_layout()

