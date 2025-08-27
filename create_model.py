# Require imports
import json
import argparse
from PIL import Image

import numpy as np

import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
import torchvision.transforms.functional as F

import onnx
import onnxruntime as ort
from onnx import compose

# Create Module for Image Preprocessing
class ImagePreprocess(torch.nn.Module):
    def __init__(
        self
        , mean = [0.485, 0.456, 0.406]
        , std = [0.229, 0.224, 0.225]
    ):
        super(ImagePreprocess, self).__init__()
        self.normalize = transforms.Normalize(
            mean=mean,
            std=std)
        self.mean = mean
        self.std = std

    def forward(self, x):
        x = x.reshape(224,224,4)
        x = x[:,:,0:3]
        x = x.permute(2, 0, 1)
        x = x.unsqueeze(0)
        x = F.convert_image_dtype(x, torch.float)
        return x

def get_image_tensor(image_name):
    # Create example input image
    img = Image.open(f"./img/{image_name}.jpg").convert('RGBA')

    # Transform image into same format as webpage
    img = img.resize((224, 224))
    img = np.array(img)
    img = torch.from_numpy(img)
    img = img.flatten()

    return img

def create_onnx_image_preprocessing(image_tensor, onnx_file_name):
    # Export Image Preprocessing module to ONNX graph
    model_preprocess = ImagePreprocess()
    torch.onnx.export(model_preprocess
                      , image_tensor
                      , f"{onnx_file_name}.onnx"
                      , input_names=["input"]
                      , output_names=["output"])

    return model_preprocess

def validate_model_output(img_preprocessed, model, expected_output, class_labels):
    # Ensure model is in eval mode for inference
    model.eval()

    # Pass image through model
    with torch.no_grad():
        output = model(img_preprocessed)

    # Get class int with the highest probability
    predicted_class_int = torch.argmax(output).item()

    # Get class label using class int
    predicted_class_label = class_labels[predicted_class_int]

    assert predicted_class_label == expected_output, f"validate_model_output: Expected {expected_output} but got {predicted_class_label}."

def validate_onnx_model_output(image_tensor, onnx_webpage_model_name, expected_output, class_labels):
    # Load the ONNX model
    session = ort.InferenceSession(f"{onnx_webpage_model_name}.onnx"
                                   , providers=['CPUExecutionProvider'])

    # Get input and output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Run inference
    results = session.run([output_name], {input_name: image_tensor.numpy()})

    # Get output data as tensor
    output_data = torch.tensor(results[0])

    # Get class int with highest probability
    predicted_class_int = torch.argmax(output_data).item()

    # Get class label using class int
    predicted_class_label = class_labels[predicted_class_int]

    assert predicted_class_label == expected_output, f"validate_onnx_model_output: Expected {expected_output} but got {predicted_class_label}."

def create_onnx_model(image_tensor, preprocess_module, expected_output, onnx_file_name):
    # Initialize the ResNet50 weights & model
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)

    # Get class labels from ResNet50 weights
    class_labels = weights.meta["categories"]

    # Save class labels for use by webpage
    with open('class_labels.json', 'w') as json_file:
        json.dump({"class_labels": class_labels}, json_file)

    # Transform input using Image Preprocess
    img_preprocessed = preprocess_module(image_tensor)

    # Validate model output
    validate_model_output(img_preprocessed, model, expected_output, class_labels)

    # Export ResNet50 as ONNX graph
    torch.onnx.export(model
                      , img_preprocessed
                      , f"{onnx_file_name}.onnx"
                      , input_names=["input"]
                      , output_names=["output"])

    return class_labels

def create_onnx_webpage(image_tensor, expected_output, class_labels, onnx_preprocess_name, onnx_model_name, onnx_file_name):
    # Load ONNX graphs for image preprocess & resnet50 model
    prep = onnx.load(f'{onnx_preprocess_name}.onnx')
    model = onnx.load(f'{onnx_model_name}.onnx')

    # add prefix, resolve names conflits
    prep_with_prefix = compose.add_prefix(prep, prefix="prep_")

    model_prep = compose.merge_models(
        prep_with_prefix,
        model,
        io_map=[('prep_output', # output prep model
                 'input')])     # input resnet50 model

    onnx.save_model(model_prep, f'{onnx_file_name}.onnx')

    validate_onnx_model_output(image_tensor, onnx_file_name, expected_output, class_labels)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create ONNX ResNet50 model')
    parser.add_argument('--image_name')
    args = parser.parse_args()

    img_class_label = "desk" if args.image_name is None else args.image_name
    img_tensor = get_image_tensor(img_class_label)

    preprocess_module = create_onnx_image_preprocessing(img_tensor, "./onnx_exports/image_preprocess")
    class_labels = create_onnx_model(img_tensor, preprocess_module, img_class_label, "./onnx_exports/resnet50_model")
    create_onnx_webpage(img_tensor
                        , img_class_label
                        , class_labels
                        , "./onnx_exports/image_preprocess"
                        , "./onnx_exports/resnet50_model"
                        , "./onnx_exports/webpage_model")


