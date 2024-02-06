# AI Image Detector

## Introduction 🌟

Identifying whether images are AI-generated or human-made is crucial as AI's capability to produce lifelike images improves. This project is an attempt to tackle this challenge through advanced machine learning, aiming to effectively classify images with minimal uncertainty.

The AI Image Detector project endeavors to provide a tool for reliably distinguishing between AI-generated and human-created images. This is especially important in areas where the authenticity of an image is critical, such as journalism and legal matters.

## Overview and Model Information 🌐📊
![CvT-13 Model Architecture](https://github.com/microsoft/CvT/blob/main/figures/pipeline.svg)

This tool, designed around Microsoft's CvT-13 model, has been custom-tuned to better discern between AI and human image generation. Training on a dataset of 2.5 million varied images has significantly advanced its image recognition capabilities. The integration with Hugging Face's API further enhances its usability and adaptability.

### HuggingFace🤗
Coming soon...

### Key Features 🚀

- **CvT-13 Architecture**: A highly optimized version of Microsoft's Convolutional Vision Transformer.
- **Massive Dataset**: Painstakingly trained on a diverse and extensive set of 2.5 million images.
- **Custom Evaluation Metrics**: Advanced evaluation scripts for in-depth performance analysis, including confusion matrices.
- **Flexible Data Handling**: A bespoke data loader for streamlined integration and efficient processing.

## Table of Contents 

1. [Dataset](#dataset)
2. [File Structure](#file-structure)
3. [Preparing for Quick Start](preparing-for-quick-start-)
4. [Quick Start](#quick-start-)
5. [Results](#results-)
6. [Contributing](#contributing-)
7. [License](#license)
8. [Acknowledgments](#acknowledgments-)


## Dataset

The model's development leveraged a comprehensive dataset originally curated by [AWSAF](https://github.com/awsaf49/artifact), consisting of approximately 2.5 million images that include a mix of AI-generated and human-created images. To enhance the training and testing process, this dataset was further processed and organized, ensuring an efficient and effective fine-tuning phase for the CvT-13 model to achieve high accuracy and performance.

Due to the substantial size of the training and testing datasets, they are not hosted on GitHub but are made available via Google Drive for convenience:

- **Training Data**: [Train.zip](https://drive.google.com/file/d/1-1ddgedsRSvJm3ERQwJPy4tq4cB0uWe9/view?usp=sharing) **26.47 GB**
- **Testing Data**: [Test.zip](https://drive.google.com/file/d/1-1xneYPH9fgSPCVnlZrhCM6c0FpFEl6B/view?usp=sharing) **2.93 GB**
- **Train Labels**: [train.csv](https://drive.google.com/file/d/1rM2r7cxve7ApXCHTlBnMyD50n5hfnoAX/view?usp=sharing) **95.6 MB**
- **Test Labels**: [test.csv](https://drive.google.com/file/d/1-GzzsszBlrmUHaDvoqVJgFfLvzRQDaBV/view?usp=sharing) **10.2 MB**

To replicate the training and evaluation results of this model, please download the above datasets before proceeding with the setup.



## File Structure 

```
ai-image-detector/
│
├── models/
│   ├── model_epoch_24.pth
│
└── src/
    ├── custom_dataset.py
    ├── evaluate.py
    ├── main.py
    ├── model.py
    └── train.py
```
## Preparing for Quick Start 🛠️

Before diving into the Quick Start, ensure your environment is set up correctly. This includes installing required packages and navigating to the correct directory.

### Required Libraries 


![PyTorch Badge](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&style=for-the-badge&logo=PyTorch&logoColor=white)

![tqdm Badge](https://img.shields.io/badge/tqdm-%232196F3.svg?&style=for-the-badge&logo=tqdm&logoColor=white) 

![NumPy Badge](https://img.shields.io/badge/NumPy-%23013243.svg?&style=for-the-badge&logo=numpy&logoColor=white) 

![Matplotlib Badge](https://img.shields.io/badge/Matplotlib-%231F425F.svg?&style=for-the-badge&logo=matplotlib&logoColor=white) 

![scikit-learn Badge](https://img.shields.io/badge/scikit_learn-%23F7931E.svg?&style=for-the-badge&logo=scikit-learn&logoColor=white) 

![mlxtend Badge](https://img.shields.io/badge/mlxtend-%23F9A03C.svg?&style=for-the-badge&logo=mlxtend&logoColor=white) 

![Pillow Badge](https://img.shields.io/badge/Pillow-%237B68EE.svg?&style=for-the-badge&logo=pillow&logoColor=white)

### Downloading Model Weights

The model weights file is too large to be hosted on GitHub and is instead available via Google Drive. Please download the model weights before proceeding with the setup:

- **Model Weights**: [Download Here](https://drive.google.com/file/d/1Wb1z9d_Nr4nKKYaevBym684xYGpvJRhY/view?usp=sharing) **226.6 MB**

After downloading, move the file into the `ai-image-detector/models/` directory. You can do this manually or by running the following command in your terminal:

```bash
mv path/to/downloaded/model_epoch_24.pth path/to/ai-image-detector/models/
```
Replace **path/to/downloaded/model_epoch_24.pth** with the actual path to the downloaded .pth file and path/to/ai-image-detector/ with the path to your local clone of the repository.

### Installing Required Packages 

You'll need to install several Python packages to work with the AI Image Detector. You can install these packages using the following command:
```bash
pip install --upgrade transformers
```
```bash
pip install --upgrade mlxtend
```
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
```bash
pip install tqdm numpy matplotlib scikit-learn pillow
```

### Navigating to Your Project Directory 

Once the packages are installed, navigate to the directory where you've cloned or downloaded the AI Image Detector project:

```bash
cd path/to/ai-image-detector
```

Replace `path/to/ai-image-detector` with the actual path to project directory.
### Tip for Data Organization

> 💡**Tip:** When organizing your image data for classification with PyTorch's `datasets.ImageFolder`, it's important to name your directories appropriately for correct label assignment. If you want to label "real" images as 0 and "fake" images as 1, arrange your directory names in alphabetical order to match this. Use names like `A_real` and `B_fake` to ensure `datasets.ImageFolder` assigns labels 0 to "real" and 1 to "fake" respectively.

#### Example Structure:

```
data_dir/
├── A_real/  # This will be labeled as 0 by ImageFolder
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── B_fake/  # This will be labeled as 1 by ImageFolder
    ├── img1.jpg
    ├── img2.jpg
    └── ...
```

This naming convention ensures that your dataset is correctly organized for training. The labels will be automatically and accurately assigned by `ImageFolder`, which is especially useful in binary classification tasks like differentiating "real" from "fake" images.
### Setting Up Your Environment Variables 

Ensure that your Python environment is set up correctly for PyTorch. If you're using a GPU, make sure your CUDA environment is properly configured. Refer to the official PyTorch documentation for detailed instructions.

Now, you're all set to proceed with the Quick Start section of the AI Image Detector project!


## Quick Start 🚀

This section guides you through the initial steps to get up and running with the AI Image Detector. It includes procedures for data preparation, model evaluation, and making predictions with a trained model.

### Preparing Your Data 

For preparing your data for training or evaluation, ensure it's organized in a directory structure with each subdirectory representing a class. The `custom_dataset.py` script will automatically handle necessary transformations and convert your images into a PyTorch dataset.

### Evaluating the Model 

To evaluate a trained model on a test dataset, use the `evaluate.py` script. Provide the directory of your test data and, if desired, the path to the model weights folder:

```bash
python src/evaluate.py /path/to/your/test/data --weights_folder=/path/to/model/model_weights
```

If the model weights path is not specified, the script defaults to the latest model in the `./models` folder. The script outputs the model's performance metrics and displays a confusion matrix for your test data.

### Making Predictions 

For predictions on individual images, use the `main.py` script. Specify the image path and, if needed, a specific model weight file:

```bash
python src/main.py /path/to/your/image.jpg --weights_folder=/path/to/model_weight
```

Without a specified model weight path, the script defaults to the latest model in the `./models` folder. The script outputs the predicted class and associated probabilities.


### Training the Model 

Train your model using the `train.py` script. Define the directory for your training data and optionally set the number of epochs or a custom learning rate:

```bash
python src/train.py /path/to/your/training/data --total_epochs=50 --learning_rate=1e-4
```

The script resumes from the latest checkpoint and saves new checkpoints after each epoch.

### Tip for Model Saving 

> 💡**Tip:** It's recommended to save models in the `models/` directory with the naming 'model_epoch_*.pth'. This ensures smooth compatibility with the evaluation and prediction scripts, which auto-locate and load the latest model from this directory.


## Results 📈

Visualizations of the model's performance are below:

- **Training Loss Graph**:
- ![Training Loss Graph](/metrics/training_loss.png)
- **Confusion Matrix**:
- ![Confusion Matrix](/metrics/confusion_matrix.png)

The model, evaluated on test data, achieved these metrics:

| Metric            | Value     |
|-------------------|-----------|
| Average Test Loss | 0.1275    |
| Accuracy          | 98.54%    |
| Precision         | 0.99      |
| Recall            | 0.98      |
| F1 Score          | 0.98      |

## Contributing 🤝

Contributions to the AI Image Detector are warmly welcomed. Check the contributing guidelines for details on submitting pull requests.

## License 

This project is open-sourced under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Acknowledgments 🙏

- Microsoft's CvT-13 Model: [Microsoft/CvT](https://github.com/microsoft/CvT)
- Hugging Face's CvT-13 API: [huggingface.co/microsoft/cvt-13](https://huggingface.co/microsoft/cvt-13)
- Dataset by [AWSAF](https://github.com/awsaf49/artifact)


