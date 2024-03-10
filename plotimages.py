import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

def yolo2bbox(bbox):
    x_min, y_min = bbox[0] - bbox[2] / 2, bbox[1] - bbox[3] / 2
    x_max, y_max = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
    return x_min, y_min, x_max, y_max


def plot_box(image, bbox, labels):
    h, w, d = image.shape
    for box_idx, box in enumerate(bbox):
        # x1, y1, x2, y2 = yolo2bbox(box)
        x1, y1, x2, y2 = box
        x_min = int(x1 * w)
        y_min = int(y1 * h)
        x_max = int(x2 * w)
        y_max = int(y2 * h)

        linewidth = max(2, int(w / 300))
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(255, 0, 0), thickness=linewidth)

    return image


def plot(imagepath, labelpath, num_samples):
    imagelist = []
    imagelist.extend(glob(imagepath + "/*.jpg"))
    imagelist.extend(glob(imagepath + "/*.JPG"))
    imagelist.extend(glob(imagepath + "/*.png"))
    imagelist.extend(glob(imagepath + "/*.PNG"))

    imagelist.sort()
    
    imagechosen = np.random.choice(imagelist, size=num_samples, replace=False)
    labelchosen = np.array([s[:-3].replace("images", "labels") + "txt" for s in imagechosen])
    
    plt.figure(figsize=(15, 10))
    for i, (i_pth, l_pth) in enumerate(zip(imagechosen, labelchosen)):
        image = cv2.imread(i_pth)
        with open(l_pth, "r") as f:
            bboxes = []
            labels = []
            lines = f.readlines()
            for L in lines:
                line_input = L.split()
                labels.append(line_input[0])
                x_c, y_c, w, h = [float(num) for num in line_input[1:]]
                bboxes.append(yolo2bbox([x_c, y_c, w, h]))

            print(labels)
            annotated_image = plot_box(image, bboxes, labels)
            plt.subplot(2, int(num_samples / 2) + 1, i + 1)
            plt.imshow(annotated_image)
            plt.axis("off")

    plt.subplots_adjust(wspace=0.1)
    plt.show()