import numpy as np
import pickle, cv2, os

images_dir = "data"

gt = pickle.load(open('gt_pascal.pkl', 'rb'))
keys = list(gt.keys())
key = keys[0]
print(key, gt[key])
im = cv2.imread(os.path.join(images_dir, key))
for i in range(gt[key].shape[0]):
	xmin, ymin, xmax, ymax = gt[key][i][:4]
	xmin = int(xmin*im.shape[1])
	ymin = int(ymin*im.shape[0])
	xmax = int(xmax*im.shape[1])
	ymax = int(ymax*im.shape[0])
	cv2.rectangle(im, (xmin, ymin), (xmax, ymax), (255,0,0), 2)
cv2.imwrite("output"+key, im)