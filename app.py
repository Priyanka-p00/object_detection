import os
from flask import Flask, request, render_template, send_from_directory
import cv2
import numpy as np
import torchvision
import torchvision.transforms as transforms

app = Flask(__name__)

# COCO dataset class names
classes = [
    'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack',
    'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk',
    'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'hair brush'
]

# load the pytorch model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# set the model in evaluation mode
model.eval()

# transform to apply to input images
transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        image_path = os.path.join('uploads', uploaded_file.filename)
        uploaded_file.save(image_path)
        annotated_image = detect_objects(image_path)
        return render_template('result.html', image_file=image_path, annotated_image=annotated_image)
    return render_template('upload.html', message='No file selected')

def detect_objects(image_path):
    # Read the image file
    img = cv2.imread(image_path)

    # Transform the input to tensor
    nn_input = transform(img)
    output = model([nn_input])

    # Random color for each class
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # Iterate over the network output for all boxes
    for box, box_class, score in zip(output[0]['boxes'].detach().numpy(),
                                      output[0]['labels'].detach().numpy(),
                                      output[0]['scores'].detach().numpy()):
        # Filter the boxes by score
        if score > 0.5:
            # Transform bounding box format
            box = [(int(box[0]), int(box[1])), (int(box[2]), int(box[3]))]

            # Select class color
            color = colors[box_class]

            # Extract class name
            class_name = classes[box_class]

            # Draw the bounding box
            cv2.rectangle(img=img,
                          pt1=box[0],
                          pt2=box[1],
                          color=color,
                          thickness=2)

            # Display the box class label
            cv2.putText(img=img,
                        text=class_name,
                        org=box[0],
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=color,
                        thickness=2)

    annotated_image_path = os.path.join('annotated_images', 'annotated_' + os.path.basename(image_path))
    cv2.imwrite(annotated_image_path, img)
    return annotated_image_path

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory('uploads', filename)

@app.route('/annotated_images/<filename>')
def send_annotated_file(filename):
    return send_from_directory('annotated_images', filename)

if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    if not os.path.exists('annotated_images'):
        os.makedirs('annotated_images')
    app.run(debug=True)
