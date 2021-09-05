import glob
import xml.etree.ElementTree as ET

import numpy as np

from kmeans import kmeans, avg_iou

ANNOTATIONS_PATH = "../annotations"
CLUSTERS = 9
SIZE = 640

def load_dataset(path):
	dataset = []
	for xml_file in glob.glob("{}/*xml".format(path)):
		tree = ET.parse(xml_file)

		height = SIZE
		width = SIZE

		for obj in tree.iter("object"):
			xmin = int(eval(obj.findtext("bndbox/xmin"))) / width
			ymin = int(eval(obj.findtext("bndbox/ymin"))) / height
			xmax = int(eval(obj.findtext("bndbox/xmax"))) / width
			ymax = int(eval(obj.findtext("bndbox/ymax"))) / height

			if (xmax - xmin) * (ymax - ymin) == 0:
				continue
			dataset.append([xmax - xmin, ymax - ymin])

	print(len(dataset))
	return np.array(dataset)


data = load_dataset(ANNOTATIONS_PATH)
out = kmeans(data, k=CLUSTERS)
print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
print("Boxes:\n {}".format(out * SIZE))

ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
print("Ratios:\n {}".format(sorted(ratios)))