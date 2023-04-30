import torch
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os 
import argparse

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   


parser = argparse.ArgumentParser()

parser.add_argument('--image-path', type=str, default='./results/', help='the path to load the image')
parser.add_argument('--save-dir', type=str, default='./logs/', help='the path to log the results')
parser.add_argument('--name', type=str, default='1.png', help='the path to log the results')
parser.add_argument('--x', type=int, default=50, help='the path to log the results')
parser.add_argument('--y', type=int, default=50, help='the path to log the results')


args = parser.parse_args()

image = cv2.imread(args.image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print("Image Size: ", image.shape)

###Put the point here
input_point = np.array([[args.x, args.y]])
input_label = np.array([1])

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

plt.figure(figsize=(10,10))
plt.imshow(image)
show_points(input_point, input_label, plt.gca())
plt.axis('on')
plt.savefig(os.path.join(args.save_dir, args.name))
plt.show()  
