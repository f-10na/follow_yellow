import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import cv2

class RopeDataset(Dataset):
    def __init__(self, colour_path, crop_function, detect_function, extract_function):
        self.samples = []

        cap = cv2.VideoCapture(colour_path)

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),  # ResNet expects 224x224
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                                 std=[0.229, 0.224, 0.225])
        ])

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            cropped = crop_function(frame)
            mask = detect_function(cropped)
            e, k = extract_function(mask, cropped.shape[1])

            if e is None:
                continue

            tensor = transform(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            self.samples.append((tensor, torch.tensor([e, k], dtype=torch.float32)))

        cap.release()
        print(f'Dataset size: {len(self.samples)} frames')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
