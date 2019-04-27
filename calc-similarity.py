from glob import glob
import sys
import numpy as np
import cv2
# from PIL import Image, ImageDraw, ImageFont
#import matplotlib.pyplot as plt

def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return np.array(img).astype(np.int)

def save_img(path, img):
    return cv2.imwrite(path, img)

def resize_img(img, size):
    return cv2.resize(img, dsize=size, interpolation=cv2.INTER_LINEAR)

def normalize_img(img):
    img = np.array(img).astype(np.float)
    probmap = img/255;
    return probmap

def binarize_probmap(probmap, threshold=0.5):
    b_probmap = np.zeros_like(probmap).astype(np.int)
    b_probmap[probmap>threshold] = 1
    return b_probmap

def calcscore(imgPred, imgTarget, threshold=threshold, method="dice"):
    ## function takes two images and returns a similarity score
    ## shapes of input images must be identical
    ## threshold is used to binarize image to binary maps
    ## method specifies which formula to evaluate similarity

    # normalize imgs into probability maps
    probmapPred = normalize_img(imgPred)
    probmapTarget = normalize_img(imgTarget)

    # binarize probmap
    probmapPred = binarize_probmap(probmapPred, threshold=threshold)
    probmapTarget = binarize_probmap(probmapTarget, threshold=threshold)

    # calculate similarity score
    TP = np.sum(probmapPred&probmapTarget)
    FP = np.sum((probmapPred^probmapTarget)&probmapPred)
    FN = np.sum((probmapPred^probmapTarget)&(~probmapPred))
    if method is "iou": 
        return TP / (TP + FP + FN)
    elif method is "dice": 
        return 2*TP / (2*TP + FP + FN)
    else:
        sys.exit("[!] undefined type of scoring method.")
        return None

# parse command line arguments
parser = argparse.ArgumentParser(description='')
parser.add_argument('--method', dest='method', default='dice', help='type of index for similarity')
parser.add_argument('--saveplot', dest='saveplot', default=True, help='save plot image')
parser.add_argument('--savecsv', dest='savecsv', default=True, help='save csv file')
args = parser.parse_args()

## Load images
# specify image directory
pred_files = glob("./pred/*.png")
target_files = glob("./target/*.png")

# check if same number of files are found
if len(pred_files)==len(target_files):
	numpairs = len(pred_files) # number of pairs
	print("[*] found %d image pairs." % numpairs)
else
	sys.exit("[!] number of files do not match.")

## Calculate score of image pairs
list_score = [] # list of scores

for i, (pred_dir, target_dir) in enumerate(pred_files, target_files):

    # load image
    pred_img = load_img(pred_dir)
    target_img = load_img(target_dir)

    # resize imgs to 256x256 pixels
    pred_img = resize_img(pred_img, [256, 256])
    target_img = resize_img(target_img, [256, 256])

    # calculate dice score
    score = calcscore(pred_img, target_img, threshold=0.5, method=args.method)
    print("#%d score: %.4f" % (i+1, score))
    list_score.append(score)

    # save plot to PNG
    if args.saveplot is True
        plot = 0.5*pred_img + 0.5*target_img
        # draw = ImageDraw.Draw(plot)
        # draw.text(xy=(0,0), text=str(score), fill=256)
        # plot.show()
        save_img(str(i)+".png", plot)

# print average score
print("avg score: %0.4f" % np.average(list_score))

# save to CSV
if args.savecsv is True
    np.savetxt("scores.csv", list_score, delimiter=",")