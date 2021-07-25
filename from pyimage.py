import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import measure, color, io
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import exposure
from skimage import data, img_as_float
kernel2 = np.ones((5, 5), np.float)/9
kernel3 = np.ones((7, 7), np.float)/9
kernel = np.ones((3, 3), np.uint8)
img1= cv2.imread("image/lemon.jpg", cv2.IMREAD_COLOR)
img1 = cv2.resize(img1,(500,500))
# img_Blur = cv2.filter2D(img1,-1,kernel2)
# cv2.imshow("image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img,(500,500))

# sigma_est = np.mean(estimate_sigma(img,multichannel=True))
# median = denoise_nl_means(img, h=1.15 * sigma_est, fast_mode=False,patch_size=5,patch_distance=6,multichannel=True, preserve_range=True)
# median = exposure.equalize_hist(median)
#

# # img = exposure.equalize_adapthist(img, kernel_size=None, clip_limit=0.01, nbins=256)
# # img = cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
ret1, thresh = cv2.threshold(img,170, 255, cv2.THRESH_BINARY)
ret2, thresh2 = cv2.threshold(img,200, 255, cv2.THRESH_BINARY)
# thresh = cv2.filter2D(thresh,-1,kernel3)
bg = cv2.dilate(thresh, kernel, iterations=3)
dist_transform = cv2.distanceTransform(bg, cv2.DIST_L2, 3)
ret3, fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)
# # ret2, thresh2 = cv2.threshold(img,100, 255, cv2.THRESH_BINARY)
# # thresh2 = cv2.erode(thresh2, kernel, iterations=2)
#
unknown = cv2.subtract(bg,thresh2)
#
ret4, markers = cv2.connectedComponents(thresh2)
markers = markers+1
markers [unknown ==255] = 0
# markers = cv2.normalize(src=markers , dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
# img1= cv2.normalize(src=img1 , dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
plt.imshow(markers , cmap = "jet")
plt.hist(img.flat,bins =100 , range =(0,255))
plt.show()
#
markers = cv2.watershed(img1,markers)
img1[markers ==-1] = [255,0,255]
img2 = color.label2rgb(markers)
#
# fg= cv2.normalize(src=fg, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
# #
# # #dilate = cv2.dilate(thresh, kernel, iterations=4)
# # erode = cv2.erode(thresh, kernel, iterations=1)
# # canny = cv2.Canny(img,50,50)
# # bg = cv2.bitwise_not(canny)
# #
# # # bg = cv2.bitwise_not(erode)
# # # unknown =cv2.bitwise_xor(thresh2,canny )
# # # e_unknown = cv2.erode(unknown, kernel, iterations=3)
# # # fg = cv2.subtract(bg,e_unknown)
# # # fg =cv2.bitwise_and(thresh2,fg )
# # # dist_transform = cv2.distanceTransform(fg, cv2.DIST_L2, 3)
# # # ret3, fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)
# # #
#plt.hist(median.flat,bins =100 , range =(0,255))
#plt.show()
cv2.imshow("image", img)
# cv2.imshow("median", median)
cv2.imshow("thresh", thresh)
cv2.imshow("thresh2", thresh2)
cv2.imshow("bg", bg)
cv2.imshow("fg", fg)
# cv2.imshow("thresh", thresh)
# # cv2.imshow("thresh 2", thresh2)
# # cv2.imshow("erode", erode)
# cv2.imshow("canny", canny)
# # cv2.imshow("unknown", unknown)
# cv2.imshow("fg", fg)
# # cv2.imshow("eroded unknown", e_unknown)
cv2.imshow("original image",img1)
cv2.imshow("Colored grains",img2)
# # #cv2.imshow("dilate",dilate)
# cv2.imshow("bg", bg)
cv2.waitKey(0)
cv2.destroyAllWindows()