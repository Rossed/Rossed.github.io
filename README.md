# 🤖 Model Integration Demo
> Demo project showcasing a [ResNet50](https://blog.roboflow.com/what-is-resnet-50/) classification model in [ONNX](https://onnx.ai/) format running inference in the browser.

There are two main parts to this project:
1. [create_model.py](./create_model.py) - Python script to create the required onnx model from a pre-trained [pytorch ResNet50 model](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html). The pytorch model is pre-trained on the 1000 class [ImageNet dataset](https://www.image-net.org/about.php).
2. [index.html](./index.html) - A simple demo webpage that uses the model created by [create_model.py](./create_model.py) to run inference. The webpage provides four images, that when clicked, are passed through the classification model and the result is displayed back to the user.  

## Creating the ONNX Model

Run [create_model.py](./create_model.py)

```shell
    python create_model.py --image_name {image_net_class_name}
```
Note: 
1. The *--image_name* argument needs to be one of the ImageNet class names (e.g. desk, stingray, etc.). In addition, there must be an associated image of the same name in the [img](./img) directory (e.g. For *--input_image desk* there must be a *./img/desk.jpg* in the ./img directory). This image is used as a dummy input for creation of the ONNX model.
2. ONNX Models will be exported to the [./onnx_exports](./onnx_exports) directory.

## ONNX Model Inference Webpage
The [webpage](./index.html) is hosted as a GitHub Pages site and can be viewed here: [Model Integration Demo](https://rossed.github.io/)