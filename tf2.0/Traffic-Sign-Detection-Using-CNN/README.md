# Traffic-Sign-Detection-Using-CNN

This project is a Traffic Sign Detection application developed using TensorFlow and Keras libraries. The model was trained using a Convolutional Neural Network (CNN) on the [Traffic Sign Dataset for Classification](https://www.kaggle.com/datasets/ahemateja19bec1025/traffic-sign-dataset-classification) from Kaggle. The goal of this project is to accurately identify and classify traffic signs from images, aiding in real-time traffic sign recognition systems.

## Convolutional Neural Networks (CNNs)
Convolutional Neural Networks (CNNs) are a class of deep neural networks particularly well-suited for image classification tasks. They are inspired by the visual cortex of animals and are designed to automatically and adaptively learn spatial hierarchies of features from input images. CNNs consist of the following key components:

### Convolutional Layers
Convolutional layers apply convolution operations to input images, extracting various features such as edges, textures, and patterns. These layers consist of learnable filters or kernels that slide across the input image, performing element-wise multiplication and summation to produce feature maps.

### Pooling Layers
Pooling layers reduce the spatial dimensions of feature maps while retaining important information. Max pooling, for example, selects the maximum value within a certain region of the feature map, effectively downsampling the input and making the network more robust to variations in input images.

### Fully Connected Layers
Fully connected layers take the high-level features extracted by convolutional and pooling layers and use them for classification tasks. These layers connect every neuron in one layer to every neuron in the next layer, allowing the network to learn complex decision boundaries and make predictions.

### Activation Functions
Activation functions introduce non-linearity into the network, enabling it to learn complex mappings between input and output. Common activation functions include ReLU (Rectified Linear Unit), which introduces sparsity and accelerates convergence, and softmax, which produces probabilities for multi-class classification tasks.

## Model Training and Accuracy
The model in this project was trained using TensorFlow and Keras, with the following specifications:
- Utilization of CNN architecture optimized for image classification tasks.
- Training on the Traffic Sign Dataset for Classification, which consists of thousands of images across various classes of traffic signs.

After training, the model achieved an impressive accuracy of **98.89%** on the test dataset, indicating its robustness and effectiveness in classifying traffic signs.

## Usage
To use the application, follow these steps:
1. Clone the repository to your local machine.
2. Install the required dependencies specified in `requirements.txt`.
3. Run the `main.py` file using Python.

Upon running `main.py`, a graphical user interface (GUI) will be launched where users can interact with the application. The GUI provides the following functionalities:
- **Upload Photo:** Users can upload their own images containing traffic signs.
- **Classify Traffic Sign:** Once an image is uploaded, users can initiate the classification process to identify the traffic sign present in the image.

## Screenshots
Below are some screenshots showcasing the application interface and functionality:

- **Screenshot 1:**
  ![1](https://github.com/MustafaBanatwala04/Traffic-Sign-Detection-Using-CNN/assets/142564605/c7db75ae-1209-4651-99fd-9600e4b3fd8c)

- **Screenshot 2:**
  ![3](https://github.com/MustafaBanatwala04/Traffic-Sign-Detection-Using-CNN/assets/142564605/deb24fb7-758d-44a2-8757-de9da3a4553f)

- **Screenshot 3:**
  ![5](https://github.com/MustafaBanatwala04/Traffic-Sign-Detection-Using-CNN/assets/142564605/f6c53b17-305f-4a2b-8e0e-895365d7eed8)

Feel free to explore the application and contribute to its enhancement and improvement.

## Contributions
Contributions to this project are welcome! If you have any suggestions, feature requests, or bug reports, please feel free to open an issue or submit a pull request.

## License
This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute the code for both non-commercial and commercial purposes.
