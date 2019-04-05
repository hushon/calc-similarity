from glob import glob
import numpy as np
from scipy.misc import imread, imsave, imresize
from PIL import Image, ImageDraw, ImageFont
#from imageio import imread
#from skimage.transform import resize
#import matplotlib.pyplot as plt

def load_img(path):
	img = imread(path, flatten = True).astype(np.int) # flatten into grayscale image
	return img

def resize_img(img, size):
	resized_img = imresize(img, size)
	return resized_img

def normalize_img(img):
	probmap = img/255;
	return probmap

def binarize_probmap(probmap, threshold=0.5):
	b_probmap = np.zeros_like(probmap).astype(np.int)
	b_probmap[probmap>threshold] = 1
	return b_probmap

def similarity_score(probmapPred, probmapTarget, method="dice"):
	# select between Dice score and IOU
	TP = np.sum(probmapPred&probmapTarget)
	FP = np.sum((probmapPred^probmapTarget)&probmapPred)
	FN = np.sum((probmapPred^probmapTarget)&(~probmapPred))
	if method is "iou": 
		score = TP / (TP + FP + FN)
	elif method is "dice": 
		score = 2*TP / (2*TP + FP + FN)
	else:
		print("[!] undefined type of scoring method")
		score = None

	return score

# parse command line arguments
parser = argparse.ArgumentParser(description='')
parser.add_argument('--method', dest='method', default='dice', help='type of index for similarity')
parser.add_argument('--saveplot', dest='saveplot', default=True, help='save plot image')
parser.add_argument('--savecsv', dest='savecsv', default=True, help='save csv file')
args = parser.parse_args()

# specify image directory
pred_files = glob("./pred/*.png")
target_files = glob("./target/*.png")

idx = range(len(pred_files)) # number of pairs
list_score = [] # save scores of each pair

for (i, pred_dir, target_dir) in zip(idx, pred_files, target_files):

	# load image
	pred_img = load_img(pred_dir)
	target_img = load_img(target_dir)

	# resize imgs to 256x256 pixels
	pred_img_256 = resize_img(pred_img, [256, 256])
	target_img_256 = resize_img(target_img, [256, 256])

	# normalize imgs into probability maps
	pred_probmap = normalize_img(pred_img_256)
	target_probmap = normalize_img(target_img_256)

	# binarize probmap
	bin_pred_probmap = binarize_probmap(pred_probmap, threshold=0.5)
	bin_target_probmap = binarize_probmap(target_probmap, threshold=0.5)

	# calculate dice score
	score = similarity_score(bin_pred_probmap, bin_target_probmap, method=args.method)
	print("#%d score: %.4f" % (i+1, score))
	list_score.append(score)

	# save plot to PNG
	if args.saveplot is True
		plot = 0.5*bin_pred_probmap + 0.5*bin_target_probmap
		# draw = ImageDraw.Draw(plot)
		# draw.text(xy=(0,0), text=str(score), fill=256)
		# plot.show()
		imsave(str(i)+".png", plot)

# print average score
print("avg score: %0.4f" % np.average(list_score))

# save to CSV
if args.savecsv is True
	np.savetxt("scores.csv", list_score, delimiter=",")