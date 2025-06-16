# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# utilitary functions about images (loading/converting...)
# --------------------------------------------------------
import os
import re
import torch
import numpy as np
import PIL.Image
from PIL.ImageOps import exif_transpose
import torchvision.transforms as tvf
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import matplotlib.pyplot as plt
import cv2  # noqa
from PIL import Image
try:
    from pillow_heif import register_heif_opener  # noqa
    register_heif_opener()
    heif_support_enabled = True
except ImportError:
    heif_support_enabled = False

ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def load_images_for_eval(
    folder_or_list, size, square_ok=False, verbose=True, crop=True
):
    """open and convert all images in a list or folder to proper input format for DUSt3R"""
    if isinstance(folder_or_list, str):
        if verbose:
            print(f">> Loading images from {folder_or_list}")
        root, folder_content = folder_or_list, sorted(os.listdir(folder_or_list))

    elif isinstance(folder_or_list, list):
        if verbose:
            print(f">> Loading a list of {len(folder_or_list)} images")
        root, folder_content = "", folder_or_list

    else:
        raise ValueError(f"bad {folder_or_list=} ({type(folder_or_list)})")

    supported_images_extensions = [".jpg", ".jpeg", ".png"]
    if heif_support_enabled:
        supported_images_extensions += [".heic", ".heif"]
    supported_images_extensions = tuple(supported_images_extensions)

    imgs = []
    for path in folder_content:
        if not path.lower().endswith(supported_images_extensions):
            continue
        img = exif_transpose(PIL.Image.open(os.path.join(root, path))).convert("RGB")
        W1, H1 = img.size
        if size == 224:
            # resize short side to 224 (then crop)
            img = _resize_pil_image(img, round(size * max(W1 / H1, H1 / W1)))
        else:
            # resize long side to 512
            img = _resize_pil_image(img, size)
        W, H = img.size
        cx, cy = W // 2, H // 2
        if size == 224:
            half = min(cx, cy)
            if crop:
                img = img.crop((cx - half, cy - half, cx + half, cy + half))
            else:  # resize
                img = img.resize((2 * half, 2 * half), PIL.Image.LANCZOS)
        else:
            halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
            if not (square_ok) and W == H:
                halfh = 3 * halfw / 4
            if crop:
                img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))
            else:  # resize
                img = img.resize((2 * halfw, 2 * halfh), PIL.Image.LANCZOS)
        W2, H2 = img.size
        if verbose:
            print(f" - adding {path} with resolution {W1}x{H1} --> {W2}x{H2}")
        imgs.append(
            dict(
                img=ImgNorm(img)[None],
                true_shape=np.int32([img.size[::-1]]),
                idx=len(imgs),
                instance=str(len(imgs)),
            )
        )

    assert imgs, "no images foud at " + root
    if verbose:
        print(f" (Found {len(imgs)} images)")
    return imgs

def img_to_arr( img ):
    if isinstance(img, str):
        img = imread_cv2(img)
    return img

def imread_cv2(path, options=cv2.IMREAD_COLOR):
    """ Open an image or a depthmap with opencv-python.
    """
    if path.endswith(('.exr', 'EXR')):
        options = cv2.IMREAD_ANYDEPTH
    img = cv2.imread(path, options)
    if img is None:
        raise IOError(f'Could not load image={path} with {options=}')
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def rgb(ftensor, true_shape=None):
    if isinstance(ftensor, list):
        return [rgb(x, true_shape=true_shape) for x in ftensor]
    if isinstance(ftensor, torch.Tensor):
        ftensor = ftensor.detach().cpu().numpy()  # H,W,3
    if ftensor.ndim == 3 and ftensor.shape[0] == 3:
        ftensor = ftensor.transpose(1, 2, 0)
    elif ftensor.ndim == 4 and ftensor.shape[1] == 3:
        ftensor = ftensor.transpose(0, 2, 3, 1)
    if true_shape is not None:
        H, W = true_shape
        ftensor = ftensor[:H, :W]
    if ftensor.dtype == np.uint8:
        img = np.float32(ftensor) / 255
    else:
        img = (ftensor * 0.5) + 0.5
    return img.clip(min=0, max=1)


def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x*long_edge_size/S)) for x in img.size)
    return img.resize(new_size, interp)

def resize_numpy_image(img, long_edge_size):
    """
    Resize the NumPy image to a specified long edge size using OpenCV.
    
    Args:
    img (numpy.ndarray): Input image with shape (H, W, C).
    long_edge_size (int): The size of the long edge after resizing.
    
    Returns:
    numpy.ndarray: The resized image.
    """
    # Get the original dimensions of the image
    h, w = img.shape[:2]
    S = max(h, w)

    # Choose interpolation method
    if S > long_edge_size:
        interp = cv2.INTER_LANCZOS4
    else:
        interp = cv2.INTER_CUBIC
    
    # Calculate the new size
    new_size = (int(round(w * long_edge_size / S)), int(round(h * long_edge_size / S)))
    
    # Resize the image
    resized_img = cv2.resize(img, new_size, interpolation=interp)
    
    return resized_img

def crop_center(img, crop_width, crop_height):
    """
    Crop the center of the image.
    
    Args:
    img (numpy.ndarray): Input image with shape (H, W) or (H, W, C).
    crop_width (int): The width of the cropped area.
    crop_height (int): The height of the cropped area.
    
    Returns:
    numpy.ndarray: The cropped image.
    """
    h, w = img.shape[:2]
    cx, cy = h // 2, w // 2
    x1 = max(cx - crop_height // 2, 0)
    x2 = min(cx + crop_height // 2, h)
    y1 = max(cy - crop_width // 2, 0)
    y2 = min(cy + crop_width // 2, w)
    
    cropped_img = img[x1:x2, y1:y2]
    
    return cropped_img


def load_images(folder_or_list, size, square_ok=False, verbose=True):
    """ open and convert all images in a list or folder to proper input format for DUSt3R
    """
    if isinstance(folder_or_list, str):
        if verbose:
            print(f'>> Loading images from {folder_or_list}')
        root, folder_content = folder_or_list, sorted(os.listdir(folder_or_list))

    elif isinstance(folder_or_list, list):
        if verbose:
            print(f'>> Loading a list of {len(folder_or_list)} images')
        root, folder_content = '', folder_or_list

    else:
        raise ValueError(f'bad {folder_or_list=} ({type(folder_or_list)})')

    supported_images_extensions = ['.jpg', '.jpeg', '.png']
    if heif_support_enabled:
        supported_images_extensions += ['.heic', '.heif']
    supported_images_extensions = tuple(supported_images_extensions)

    imgs = []
    for path in folder_content:
        if not path.lower().endswith(supported_images_extensions):
            continue
        img = exif_transpose(PIL.Image.open(os.path.join(root, path))).convert('RGB')
        W1, H1 = img.size
        if size == 224:
            # resize short side to 224 (then crop)
            img = _resize_pil_image(img, round(size * max(W1/H1, H1/W1)))
        else:
            # resize long side to 512
            img = _resize_pil_image(img, size)
        W, H = img.size
        cx, cy = W//2, H//2
        if size == 224:
            half = min(cx, cy)
            img = img.crop((cx-half, cy-half, cx+half, cy+half))
        else:
            halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
            if not (square_ok) and W == H:
                halfh = 3*halfw/4
            img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))

        W2, H2 = img.size
        if verbose:
            print(f' - adding {path} with resolution {W1}x{H1} --> {W2}x{H2}')


        imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32(
            [img.size[::-1]]), idx=len(imgs), instance=str(len(imgs))))

    assert imgs, 'no images foud at '+root
    if verbose:
        print(f' (Found {len(imgs)} images)')
    return imgs

def depth_read_bonn(filename):
    # loads depth map D from png file
    # and returns it as a numpy array
    depth_png = np.asarray(Image.open(filename))
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert np.max(depth_png) > 255
    depth = depth_png.astype(np.float64) / 5000.0
    depth[depth_png == 0] = -1.0
    return depth

def readPFM(file):
    file = open(file, 'rb')

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0:  # little-endian
        endian = '<'
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    if data.ndim == 3:
        data = data.transpose(2, 0, 1)
    return data


def depth_read_dtu(depthpath):
    depth = readPFM(depthpath) / 1000
    depth = np.nan_to_num(depth, posinf=0., neginf=0., nan=0.)
    return depth

def load_images_depth(folder_or_list, size, offset, dataset_name, square_ok=False, verbose=True):
    """ open and convert all images in a list or folder to proper input format for DUSt3R
    """
    if isinstance(folder_or_list, str):
        if verbose:
            print(f'>> Loading images from {folder_or_list}')
        root, folder_content = folder_or_list, sorted(os.listdir(folder_or_list))

    elif isinstance(folder_or_list, list):
        if verbose:
            print(f'>> Loading a list of {len(folder_or_list)} images')
        root, folder_content = '', folder_or_list

    else:
        raise ValueError(f'bad {folder_or_list=} ({type(folder_or_list)})')

    supported_images_extensions = ['.jpg', '.jpeg', '.png']
    if heif_support_enabled:
        supported_images_extensions += ['.heic', '.heif']
    supported_images_extensions = tuple(supported_images_extensions)

    if dataset_name == "bonn":
        depth_root = os.path.dirname(folder_content[0]).replace('rgb_110', 'depth_110')
        folder_depth = sorted(os.listdir(depth_root))

    if dataset_name == "tum":
        depth_root = os.path.dirname(folder_content[0]).replace('rgb_90', 'depth_90')
        folder_depth = sorted(os.listdir(depth_root))

    imgs = []
    depth_list = []
    for i, path in enumerate(folder_content):
        if not path.lower().endswith(supported_images_extensions):
            continue
        img = exif_transpose(PIL.Image.open(os.path.join(root, path))).convert('RGB')
        
        if dataset_name == "bonn":
            depth = depth_read_bonn(os.path.join(depth_root, folder_depth[offset]))

        if dataset_name == "tum":
            depth = depth_read_bonn(os.path.join(depth_root, folder_depth[offset]))

        if dataset_name == "dtu":
            print(path.replace('images', 'gt_depths').replace('.png', '.pfm'))
            depth = depth_read_dtu(path.replace('images', 'gt_depths').replace('.png', '.pfm'))
            
        W1, H1 = img.size
        if size == 224:
            # resize short side to 224 (then crop)
            img = _resize_pil_image(img, round(size * max(W1/H1, H1/W1)))
        else:
            # resize long side to 512
            img = _resize_pil_image(img, size)
        W, H = img.size
        cx, cy = W//2, H//2
        if size == 224:
            half = min(cx, cy)
            img = img.crop((cx-half, cy-half, cx+half, cy+half))
        else:
            halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
            if not (square_ok) and W == H:
                halfh = 3*halfw/4
            img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))

        W2, H2 = img.size
        if verbose:
            print(f' - adding {path} with resolution {W1}x{H1} --> {W2}x{H2}')

        # img_array = np.array(img, dtype=np.float32) / 255
        # mean=0
        # std=0.2
        # noise = np.random.normal(mean, std, img_array.shape)

        # img_noisy = np.clip((img_array + noise)*255, 0, 255).astype(np.uint8)
        # img_norm = Image.fromarray(img_noisy)

        # img_norm.save(path.replace('images' , 'noise_images'))

        # imgs.append(dict(img=ImgNorm(img_norm)[None], true_shape=np.int32(
            # [img.size[::-1]]), idx=len(imgs), instance=str(len(imgs))))
        imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32(
            [img.size[::-1]]), idx=len(imgs), instance=str(len(imgs))))
        
        depth_list.append(depth)

    assert imgs, 'no images foud at '+root
    if verbose:
        print(f' (Found {len(imgs)} images)')


    return imgs, depth_list


def pixel_to_pointcloud(depth_map, focal_length_px):
    """
    Convert a depth map to a 3D point cloud.

    Args:
    depth_map (numpy.ndarray): The input depth map with shape (H, W), where each value represents the depth at that pixel.
    focal_length_px (float): The focal length of the camera in pixels.

    Returns:
    numpy.ndarray: The resulting point cloud with shape (H, W, 3), where each point is represented by (X, Y, Z).
    """
    height, width = depth_map.shape
    cx = width / 2
    cy = height / 2

    # Create meshgrid for pixel coordinates
    u = np.arange(width)
    v = np.arange(height)
    u, v = np.meshgrid(u, v)
    
    # Convert pixel coordinates to camera coordinates
    Z = depth_map
    X = (u - cx) * Z / focal_length_px
    Y = (v - cy) * Z / focal_length_px
    
    # Stack the coordinates into a point cloud (H, W, 3)
    point_cloud = np.dstack((X, Y, Z)).astype(np.float32)
    point_cloud = normalize_pointcloud(point_cloud)
    # Optional: Filter out invalid depth values (if necessary)
    # point_cloud = point_cloud[depth_map > 0]
    #print(point_cloud)
    return point_cloud

def normalize_pointcloud(point_cloud):
    min_vals = np.min(point_cloud, axis=(0, 1))
    max_vals = np.max(point_cloud, axis=(0, 1))
    #print(min_vals, max_vals)
    normalized_point_cloud = (point_cloud - min_vals) / (max_vals - min_vals)
    return normalized_point_cloud
