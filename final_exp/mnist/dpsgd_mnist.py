import os
import time
import clip

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST

from pkg_resources import packaging
import skimage
import IPython.display
from PIL import Image
import numpy as np

from collections import OrderedDict

from tqdm import tqdm

from dpsgd_optimizer import DPSGD
import sys

import tensorflow_privacy

from tensorflow_privacy.privacy.analysis import compute_noise_from_budget_lib


class PrivateModel:
    def __init__(self, batch_size=128, lr=1e-7, momentum=0.9, weight_decay=0.01, num_epochs=10, description = True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size

        self.description = description

        model, preprocess = clip.load("ViT-B/32")
        model.eval()
        self.model = model.to(self.device)
        self.preprocess = preprocess

        mnist = MNIST(os.path.expanduser("~/.cache"), transform=preprocess, download=True)
        self.classes = mnist.classes

        self.loss_img = nn.CrossEntropyLoss()
        self.loss_txt = nn.CrossEntropyLoss()

        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs

        print("Dataset: MNIST")
        print(f"Device: {self.device}")
        print(f"Batch Size: {batch_size}")
        print(f"Optimizer Parameters: lr={lr}, momentum={momentum}, weight_decay={weight_decay}")
        print(f"Classes: {self.classes}")

        self.losses = []


    def load_data(self):
        training_data = datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=self.preprocess
        )

        test_data = datasets.MNIST(
            root="data",
            train=False,
            download=True,
            transform=self.preprocess
        )
        self.training_size=len(training_data)
        self.testing_size=len(test_data)

        train_dataloader = DataLoader(training_data, batch_size=self.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_data, batch_size=self.batch_size, shuffle=True)
        
        return train_dataloader, test_dataloader
    

    def set_dp_params(self, epsilon=10, C=1.0):
        self.epsilon = epsilon
        self.delta = 1/2/self.training_size
        self.C = C

        print("Epsilon: ", self.epsilon)
        print("Delta: ", self.delta)
        print("Clip Param C: ", self.C)

        
        noise_param = compute_noise_from_budget_lib.compute_noise(n=self.training_size,
                                                    batch_size=self.batch_size,
                                                    target_epsilon=self.epsilon,
                                                    epochs=self.num_epochs,
                                                    delta=self.delta,
                                                    noise_lbd=1e-5)
        self.noise_scale = noise_param
        
        # self.noise_scale = compute_noise_params(self.C, data_size, compute_sigma_eff(self.epsilon, self.delta, self.batch_size, data_size))
        print("Noise Scale: ", self.noise_scale)

        self.optimizer = DPSGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay, noise_scale=self.noise_scale, norm_bound=self.C)


    def zeroshot_classifier(self):
        with torch.no_grad():
            zeroshot_weights = []
            if self.description:
                template = ['a photo of the number: "{}".',]
            else:
                template = [' "{}".',]
            for classname in tqdm(self.classes):
                texts = [t.format(classname) for t in template] # format with class
                texts = clip.tokenize(texts).to(self.device) # tokenize
                class_embeddings = self.model.encode_text(texts) # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(self.device)
        return zeroshot_weights
    

    def test(self, test_dataloader):
        correct_num = torch.tensor(0).to(self.device)

        text_features = self.zeroshot_classifier()
        
        for image, label in tqdm(test_dataloader):
            with torch.no_grad():
                features = self.model.encode_image(image.to(self.device))
                features /= features.norm(dim=-1, keepdim=True)
                
                similarity = (100.0 * features @ text_features)
                probs = similarity.softmax(dim=-1)

                _, pred = torch.max(probs, 1)
                num = torch.sum(pred==label.to(self.device))

                correct_num = correct_num + num

        print ('Accuracy Rate: {}'.format(correct_num/len(test_dataloader)/self.batch_size))
        # wandb.log({'test_acc': correct_num/len(test_dataloader)/self.batch_size})


    def setup_wandb(self):
        # Wandb setup for logging
        wandb.init(project="fashion-mnist")
        config = wandb.config
        config.batch_size = self.batch_size
        config.classes = self.classes

        config.lr = self.lr
        config.betas = self.betas
        config.eps = self.eps
        config.weight_decay = self.weight_decay

        config.norm_bound = self.C
        config.epsilon = self.epsilon
        config.delta = self.delta

        config.dataset = "MNIST"
        config.optimizer = "DPAdam"
    

    def train(self, train_dataloader, test_dataloader, loss_report_freq=1000):
        print(f"Num Epochs: {self.num_epochs}")
        
        batch_ct = 1
        start_training_time=time.time()
        device = self.device

        for epoch in range(self.num_epochs):
            tqdm_object = tqdm(train_dataloader, total=len(train_dataloader))
            for batch in tqdm_object:
                self.optimizer.zero_grad()

                images,texts = batch 

                images= images.to(device)
                texts = texts.to(device)
                if self.description:
                    text_tokens = clip.tokenize([ f'a photo of the number: "{self.classes[desc]}".' for desc in texts]).to(device)
                else:
                    text_tokens = clip.tokenize([  self.classes[desc] for desc in texts]).to(device)

                logits_per_image, logits_per_text = self.model(images, text_tokens)

                ground_truth = torch.arange(len(images),dtype=torch.long,device=device) # assigning labels to 

                total_loss = (self.loss_img(logits_per_image,ground_truth) + self.loss_txt(logits_per_text,ground_truth))/2
                total_loss.backward()
                
                self.optimizer.step()

                if batch_ct%loss_report_freq == 0:
                    print(total_loss)
                batch_ct += 1
            
            if epoch % 10 == 0 :
                print(f"****the {epoch}^th epoch *****")
                print("**** on training set *****")
                self.test(train_dataloader)
                print("*************************")
                print("**** on testing set *****")
                self.test(test_dataloader)
                print("*************************")
            
            # wandb.log({'val_loss': total_loss.item()})
            self.losses.append(total_loss.item())
        
        ending_training_time=time.time()
        print("Training Time: ", ending_training_time-start_training_time)
    

    def save_model(self, loss, checkpoint_name):
        torch.save(
          {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
          }, 
          checkpoint_name
        )


    def plot_losses(self, loss_plot_name="loss_plot.png"):
        print(self.losses)
        plt.plot(self.losses)
        plt.savefig(loss_plot_name)
        plt.show()


def train_wrapper():
    private_model = PrivateModel(batch_size = 32, lr = 1e-5, weight_decay=1e-2, num_epochs = 30, description= True)
    train_dataloader, test_dataloader = private_model.load_data()

    private_model.set_dp_params(epsilon=float(sys.argv[1]), C = 0.1)
    print("**********")
    private_model.train(train_dataloader, test_dataloader)
    private_model.test(test_dataloader)
    print("------------------------------------")

train_wrapper()