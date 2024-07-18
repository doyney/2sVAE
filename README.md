Contents

Download UTKFace Dataset https://susanqq.github.io/UTKFace/
aligned and cropped zip file and put in the data folder

You can download the pre-trained model from the following link:

You can use the wieghts and biases demo here:

Weights and biases
demo.ipynb - local version of gradio demo. 

vaesweep.py - runs Weights and Biases hyperparameter sweep on the main VAE

vaetrain.py - Trains the main VAE which chosen hyperparameters

2sVAE_UTKFace.ipynb - Loads the trained main VAE model. Logs normal sampled, dataset, and reconstruction images to Weights and Biases.
Creates attribute dataset, logs image of attribute dataset, trains 2s-VAE and logs generated images.

Subjective Evaluation Results.pdf - Shows all generated images for the 8 subjective evaluation tests of the two stage VAE generations.

Supporting Material.pdf - Contains link to google drive to download large files necessary to run the code

Please note:
- Outputs should be saved within jupyter notebooks for reference
- Sections of the code which require my model and the dataset will not work if run as I could not submit my trained model and the .tar file for my image dataset, as they were both >>50MB.
- My Hugging Face demo is available at https://huggingface.co/spaces/doyney/2sVAE_faces and accurately represents my code. It is a mirror of the demo.ipynb notebook.


To use the code, I have provided a link to my google drive, which contains the UTKFace dataset tar file, "UTKFace.tar.gz".
- Ensure this is unzipped within the 'data' directory

Also, the pretrained model is provided in the google drive. Please copy the 'final_model' directory into the "my_model/bestmodel/" directory