import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import random_split
from pythae.models import VAE, VAEConfig
from pythae.models.nn import BaseEncoder, BaseDecoder
from pythae.models.base.base_utils import ModelOutput
from pythae.trainers.training_callbacks import WandbCallback
from pythae.data.datasets import DatasetOutput
from pythae.trainers import BaseTrainerConfig
from pythae.pipelines.training import TrainingPipeline

class UTKFaceDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root=root, transform=transform, target_transform=target_transform)

    def __getitem__(self, index):
        X, _ = super().__getitem__(index)
        return DatasetOutput(data=X)

class UTKFace_Encoder(BaseEncoder):
    def __init__(self, lat_dim):
        super(UTKFace_Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.fc_mu = nn.Linear(128 * 25 * 25, lat_dim)
        self.fc_logvar = nn.Linear(128 * 25 * 25, lat_dim)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        output = ModelOutput(
            embedding=mu,  # mean
            log_covariance=logvar  # log variance
        )
        return output

class UTKFace_Decoder(BaseDecoder):
    def __init__(self, lat_dim):
        super(UTKFace_Decoder, self).__init__()
        self.fc = nn.Linear(lat_dim, 128 * 25 * 25)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        x = self.fc(x)
        x = x.view(-1, 128, 25, 25)  # reshape tensor
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))
        output = ModelOutput(
            reconstruction=x
        )
        return output


def train():

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"CUDA is available. GPU Name: {gpu_name}")
    elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Training on MPS device")
    else:
        device = torch.device("cpu")
        print("MPS not available, training on CPU")

    ### CHOOSE FINAL HYPERPARAMETERS FOR TRAINING
    latent_dim = 120
    batch_size = 8
    num_epochs = 15
    learning_rate = 0.000575

    transform = transforms.Compose([transforms.ToTensor(),])
    all_dataset = UTKFaceDataset(root="./data", transform=transform)
    train_size = int(0.8 * len(all_dataset))
    eval_size = len(all_dataset) - train_size
    train_dataset, eval_dataset = random_split(all_dataset, [train_size, eval_size])

    encoder = UTKFace_Encoder(latent_dim)
    decoder = UTKFace_Decoder(latent_dim)
    model_config = VAEConfig(input_dim=(3, 200, 200), latent_dim=latent_dim)
    model = VAE(model_config=model_config, encoder=encoder, decoder=decoder)

    config = BaseTrainerConfig(
        output_dir='my_model',
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_epochs=num_epochs,
    )

    # WandbCallback
    callbacks = []
    wandb_cb = WandbCallback()
    wandb_cb.setup(training_config=config, model_config=model_config, project_name="VAE_UTKFACE", entity_name="charlesdoyne")
    callbacks.append(wandb_cb)

    # Training
    pipeline = TrainingPipeline(training_config=config, model=model)
    pipeline(train_data=train_dataset, eval_data=eval_dataset, callbacks=callbacks)

if __name__ == "__main__":
    train()
