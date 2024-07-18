Contents

Google drive link:
https://drive.google.com/drive/folders/16pqpoYX9_858OAa3bLtQohJ8UeZWcjg8

To use the code, I have provided a link to my google drive, which contains the UTKFace dataset tar file, "UTKFace.tar.gz".
- Ensure this is unzipped within the 'data' directory

Also, the pretrained model is provided in the google drive. Please copy the 'final_model' directory into the "my_model/bestmodel/" directory

You can use the demo here:
https://huggingface.co/spaces/doyney/2sVAE_faces


vaetrain.py - Trains the main VAE with chosen hyperparameters

vaesweep.py - runs Weights and Biases hyperparameter sweep on the main VAE

2sVAE_UTKFace.ipynb - Loads the trained main VAE model. Logs normal sampled, dataset, and reconstruction images to Weights and Biases. Creates attribute dataset, logs image of attribute dataset, trains 2s-VAE and logs generated images.

demo.ipynb - local version of demo
