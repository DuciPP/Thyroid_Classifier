from torch.utils.data import Dataset, Subset
import xml.etree.ElementTree as ET
from PIL import Image
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import RandomSampler

def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    tirads = root.find(path="tirads")

    return tirads

def shuffle(dataset):
    return Subset(dataset, list(RandomSampler(dataset)))
    
    
class CustomDataset(Dataset):
    
    def __init__(self, data):
        self.data_len = pd.read_csv(data).shape[0]
        self.images = pd.read_csv(data).iloc[:, 0]
        self.labels = pd.read_csv(data).iloc[:, 1]
        self.classes = {None: 0,
                        '2': 1, 
                        '3': 2,    
                        '4a': 3,   
                        '4c': 4, 
                        '4b': 5,
                        '5': 6 }


    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        try:
            label = parse_xml(self.labels[index]).text
        except AttributeError:
            label = None
        
        label = self.classes[label]
        
        
        image = Image.open(self.images[index])
            
        transform = transforms.Compose([transforms.Grayscale(),
                                        transforms.RandomHorizontalFlip(),              
                                        transforms.RandomRotation(30),                  
                                        transforms.RandomVerticalFlip(),
                                        transforms.RandomRotation(140),
                                        transforms.ToTensor()])
                                        
        image = transform(image)

        return image, label