import numpy as np
import os, codecs, cv2, pickle

images_dir = "../data/"

data = dict()
filenames = os.listdir(images_dir)
for filename in filenames:
	if filename.endswith(".txt"):
		image_name = filename.replace(".txt", ".jpg")
		image = cv2.imread(os.path.join(images_dir, image_name))
		height, width, _ = image.shape
		with codecs.open(os.path.join(images_dir, filename), "r", "utf-8") as f:
			lines = f.readlines()
		if len(lines)==0: continue
		dd = []
		for line in lines:
			a = line.split(",")
			xmin = float(a[1])/width
			ymin = float(a[0])/height
			xmax = float(a[5])/width
			ymax = float(a[4])/height
			dd.append([xmin, ymin, xmax, ymax, 1])
		data[image_name]=np.array(dd)
pickle.dump(data ,open('gt_pascal.pkl','wb'))