import base64
from io import BytesIO
from torchvision.transforms import ToPILImage
import torch

class Visualizer:
    @staticmethod
    def tensor_to_base64(tensor):
        """
        Pretvara PyTorch tenzor (C, H, W) u Base64 string spreman za HTML.
        """
        tensor = tensor.cpu().clone()
        
        if len(tensor.shape) == 4:
            tensor = tensor.squeeze(0)
            
        # Pretvorba u PIL Image
        to_pil = ToPILImage()
        img = to_pil(tensor)
        
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        
        # Encode u base64
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str