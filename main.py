import torch
import torch.nn as nn
from torchvision import models, transforms
from datasets import load_dataset
from PIL import Image
import faiss
import numpy as np
import time
from google.colab import files
import matplotlib.pyplot as plt

# 1. Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Feature Extractor Class
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*(list(resnet.children())[:-1]))
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, img):
        if img.mode != 'RGB':
            img = img.convert('RGB')
        tensor = self.preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feature = self.backbone(tensor)
        return feature.flatten().cpu().numpy()

# 3. Load Dataset & Initialize Model
dataset = load_dataset("sidd707/jewelry-design-dataset", split="train")
extractor = FeatureExtractor().to(device)
extractor.eval()

# 4. Indexing Phase
embeddings = []
start_time = time.time()

print(f"Starting feature extraction on {device}...")
for i in range(len(dataset)):
    img = dataset[i]['image']
    embeddings.append(extractor(img))

embeddings = np.array(embeddings).astype('float32')
indexing_time = time.time() - start_time

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print(f"Indexing Complete. Total Time: {indexing_time:.2f}s")

# 5. Search Function
def run_visual_search():
    uploaded = files.upload()
    for filename in uploaded.keys():
        query_img = Image.open(filename)
        query_vector = extractor(query_img).reshape(1, -1).astype('float32')
        distances, indices = index.search(query_vector, 5)
        
        fig, ax = plt.subplots(1, 6, figsize=(20, 5))
        ax[0].imshow(query_img); ax[0].set_title("Query"); ax[0].axis('off')
        
        for i, idx in enumerate(indices[0]):
            result_img = dataset[int(idx)]['image']
            ax[i+1].imshow(result_img)
            ax[i+1].set_title(f"Dist: {distances[0][i]:.2f}")
            ax[i+1].axis('off')
        plt.show()

if __name__ == "__main__":
    run_visual_search()
