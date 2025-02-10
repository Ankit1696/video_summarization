import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


class Rescale:
    def __init__(self, *output_size):
        self.output_size = output_size

    def __call__(self, image):
        new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = image.resize((new_w, new_h), resample=Image.BILINEAR)
        return img

transform = transforms.Compose([
    Rescale(224, 224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# net = models.googlenet(pretrained=True).float().to(device)
net = models.googlenet(weights='DEFAULT').float().to(device)

net.eval()
fea_net = nn.Sequential(*list(net.children())[:-2])

def extract_features(frame):
    img = Image.fromarray(frame)
    img = transform(img)
    # img = img.unsqueeze(0).cuda()
    img = img.unsqueeze(0).to(device)
    features = fea_net(img).squeeze().detach().cpu()
    return features