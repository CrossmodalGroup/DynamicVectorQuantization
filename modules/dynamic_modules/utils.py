import torch
import importlib
import torchvision
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from einops import rearrange
import numpy as np

transform_PIL = transforms.Compose([transforms.ToPILImage()])

color_dict = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "white": (255, 255, 255),
    "yellow": (255, 255, 0),
    "blue": (5, 39, 175),
}

# same function in torchvision.utils.save_image(normalize=True)
def image_normalize(tensor, value_range=None, scale_each=False):
    tensor = tensor.clone()
    def norm_ip(img, low, high):
        img.clamp_(min=low, max=high)
        img.sub_(low).div_(max(high - low, 1e-5))

    def norm_range(t, value_range):
        if value_range is not None:
            norm_ip(t, value_range[0], value_range[1])
        else:
            norm_ip(t, float(t.min()), float(t.max()))

    if scale_each is True:
        for t in tensor:  # loop over mini-batch dimension
            norm_range(t, value_range)
    else:
        norm_range(tensor, value_range)
    
    return tensor

def draw_dual_grain_256res_color(images=None, indices=None, low_color="blue", high_color="red", scaler=0.9):
    # indices: [batch_size, height, weight]
    # 0 for coarse-grained and 1 for fine-grained
    if images is None:
        images = torch.ones(indices.size(0), 3, 256, 256)
    indices = indices.unsqueeze(1)
    size = 256 // indices.size(-1)
    indices = indices.repeat_interleave(size, dim=-1).repeat_interleave(size, dim=-2)


    bs = images.size(0)

    low = Image.new('RGB', (images.size(-2), images.size(-1)), color_dict[low_color])
    high = Image.new('RGB', (images.size(-2), images.size(-1)), color_dict[high_color])

    for i in range(bs):
        image_i_pil = transform_PIL(image_normalize(images[i]))

        score_map_i_np = rearrange(indices[i], "C H W -> H W C").cpu().detach().numpy()
        score_map_i_blend = Image.fromarray(
            np.uint8(high * score_map_i_np + low * (1 - score_map_i_np)))
        
        image_i_blend = Image.blend(image_i_pil, score_map_i_blend, scaler)

        if i == 0:
            blended_images = torchvision.transforms.functional.to_tensor(image_i_blend).unsqueeze(0)
        else:
            blended_images = torch.cat([
                blended_images, torchvision.transforms.functional.to_tensor(image_i_blend).unsqueeze(0)
            ], dim=0)
    return blended_images

def draw_dual_grain_256res(images=None, indices=None):
    # indices: [batch_size, height, weight]
    # 0 for coarse-grained and 1 for fine-grained
    if images is None:
        images = torch.ones(indices.size(0), 3, 256, 256)
    size = 256 // indices.size(1)
    for b in range(indices.size(0)): # batch_size
        for i in range(indices.size(1)):
            for j in range(indices.size(2)):
                y_min = size * i
                y_max = size * (i + 1)
                x_min = size * j
                x_max = size * (j + 1)
                images[b, :, y_min, x_min:x_max] = -1
                images[b, :, y_min:y_max, x_min] = -1
                if indices[b, i, j] == 1:
                    y_mid = y_min + size // 2
                    x_mid = x_min + size // 2
                    images[b, :, y_mid, x_min:x_max] = -1
                    images[b, :, y_min:y_max, x_mid] = -1

    # torchvision.utils.save_image(images, "test_draw_dual_grain.png")
    return images

def draw_triple_grain_256res_color(images=None, indices=None, low_color="blue", high_color="red", scaler=0.9):
    # indices: [batch_size, height, weight]
    # 0 for coarse-grained, 1 for median-grained, 2 for fine grain
    if images is None:
        images = torch.ones(indices.size(0), 3, 256, 256)
    indices = indices.unsqueeze(1)
    size = 256 // indices.size(-1)
    indices = indices.repeat_interleave(size, dim=-1).repeat_interleave(size, dim=-2)
    indices = indices / 2
    
    bs = images.size(0)

    low = Image.new('RGB', (images.size(-2), images.size(-1)), color_dict[low_color])
    high = Image.new('RGB', (images.size(-2), images.size(-1)), color_dict[high_color])

    for i in range(bs):
        image_i_pil = transform_PIL(image_normalize(images[i]))

        score_map_i_np = rearrange(indices[i], "C H W -> H W C").cpu().detach().numpy()
        score_map_i_blend = Image.fromarray(
            np.uint8(high * score_map_i_np + low * (1 - score_map_i_np)))
        
        image_i_blend = Image.blend(image_i_pil, score_map_i_blend, scaler)

        if i == 0:
            blended_images = torchvision.transforms.functional.to_tensor(image_i_blend).unsqueeze(0)
        else:
            blended_images = torch.cat([
                blended_images, torchvision.transforms.functional.to_tensor(image_i_blend).unsqueeze(0)
            ], dim=0)
    return blended_images

def draw_triple_grain_256res(images=None, indices=None):
    # indices: [batch_size, height, weight]
    # 0 for coarse-grained, 1 for median-grained, 2 for fine grain
    if images is None:
        images = torch.ones(indices.size(0), 3, 256, 256)
    size = 256 // indices.size(1)
    for b in range(indices.size(0)): # batch_size
        for i in range(indices.size(1)):
            for j in range(indices.size(2)):  # draw coarse-grain line
                y_min = size * i
                y_max = size * (i + 1)
                x_min = size * j
                x_max = size * (j + 1)
                images[b, :, y_min, x_min:x_max] = -1
                images[b, :, y_min:y_max, x_min] = -1

                if indices[b, i, j] > 0:  # draw median-grain line
                    y_mid = y_min + size // 2
                    x_mid = x_min + size // 2
                    images[b, :, y_mid, x_min:x_max] = -1
                    images[b, :, y_min:y_max, x_mid] = -1

                    if indices[b, i, j] == 2:  # draw fine-grain line
                        y_one_quarter = y_min + size // 4
                        y_three_quarter = y_max - size // 4
                        x_one_quarter = x_min + size // 4
                        x_three_quarter = x_max - size // 4
                        images[b, :, y_one_quarter, x_min:x_max] = -1
                        images[b, :, y_three_quarter, x_min:x_max] = -1
                        images[b, :, y_min:y_max, x_one_quarter] = -1
                        images[b, :, y_min:y_max, x_three_quarter] = -1

    return images

def instantiate_from_config(config):
    def get_obj_from_str(string, reload=False):
        module, cls = string.rsplit(".", 1)
        if reload:
            module_imp = importlib.import_module(module)
            importlib.reload(module_imp)
        return getattr(importlib.import_module(module, package=None), cls)
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def gumbel_softmax(logits, temperature=1, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    def gumbel_softmax_sample(logits, temperature=1, eps=1e-20):
        U = torch.rand(logits.shape).to(logits.device)
        sampled_gumbel_noise = -torch.log(-torch.log(U + eps) + eps)
        y = logits + sampled_gumbel_noise
        return F.softmax(y / temperature, dim=-1)

    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard

if __name__ == "__main__":
    test_image_path = "/home/huangmq/git_repo/AdaptiveVectorQuantization/temp/real_image.png"
    indices = torch.randint(0, 3, (4, 8, 8))  # .repeat_interleave(32, dim=-2).repeat_interleave(32, dim=-1)
    # images = draw_triple_grain_256res(indices=indices)
    # images = draw_dual_grain_256res_color(indices=indices)
    images = draw_triple_grain_256res_color(indices=indices)
    torchvision.utils.save_image(images, "temp/test_draw_triple_grain.png")