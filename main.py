import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import measure, color, io
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import exposure
from skimage import data, img_as_float

kernel = np.ones((3, 3), np.uint8)
kernel2 = np.ones((5, 5), np.float)/9
img1= cv2.imread("image/lemon.jpg", cv2.IMREAD_COLOR)
img1 = cv2.resize(img1,(500,500))
# img_Blur = cv2.filter2D(img1,-1,kernel2)
# cv2.imshow("image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img,(500,500))
# ret2, thresh2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
ret2, thresh2 = cv2.threshold(img,170, 255, cv2.THRESH_BINARY)
dist_transform = cv2.distanceTransform(thresh2, cv2.DIST_L2, 3)
ret3, sure_fg = cv2.threshold(dist_transform, 0.21 * dist_transform.max(), 255, 0)

sure_fg2 = cv2.erode(thresh2, kernel, iterations=3)
sure_fg2 = cv2.dilate(thresh2, kernel, iterations=3)
sure_fg2 = np.uint8(sure_fg)
sure_fg = sure_fg2

median =  exposure.equalize_hist(img)

median = cv2.normalize(src=median, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
# img = img_as_float(img)

# sigma_est = np.mean(estimate_sigma(img,multichannel=True))
# median = denoise_nl_means(img, h=1.15 * sigma_est, fast_mode=False,patch_size=5,patch_distance=6,multichannel=True, preserve_range=True)
# # median = exposure.equalize_hist(median)
#median = np.uint8(median)
plt.hist(median.flat,bins =100 , range =(0,255))


# plt.hist(eq_img.flat,bins =100 , range =(0,255))
#eq_img= np.uint8(eq_img)
# median = exposure.equalize_hist(median)
# median = np.uint8(median)

#sure_bg = cv2.dilate(rsure_bg,kernel,iterations=1)

#
# pixels_to_um = 0.5  # 1 pixel = 500 nm (got this from the metadata of original image)

# Threshold image to binary using OTSU. ALl thresholded pixels will be set to 255
ret1, thresh = cv2.threshold(median,210, 255, cv2.THRESH_BINARY)
print(ret1)
# sure_fg = cv2.bitwise_not(thresh )
# sure_fg= cv2.bitwise_xor(sure_fg, sure_fg2)
# sure_fg = cv2.bitwise_not(sure_fg )
# dist_transform = cv2.distanceTransform(sure_fg, cv2.DIST_L2, 3)
# ret4, sure_fg = cv2.threshold(dist_transform, 0.09 * dist_transform.max(), 255, 0)
# sure_fg = cv2.erode(sure_fg, kernel, iterations=2)
# sure_fg = cv2.dilate(sure_fg, kernel, iterations=2)
sure_fg= cv2.normalize(src=sure_fg, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
thresh = cv2.erode(thresh, kernel, iterations=1)
# imgCanny = cv2.Canny(thresh,100,100)
sure_bg =cv2.bitwise_not(thresh)
sure_fg =cv2.bitwise_and(sure_bg ,sure_fg)
sure_fg = cv2.erode(sure_fg , kernel, iterations=3)
#sure_fg = cv2.dilate(sure_fg , kernel, iterations=3)

# Morphological operations to remove small noise - opening
# To remove holes we can use closing

# sure_fg = cv2.dilate(thresh2, kernel, iterations=3)
# sure_fg = np.uint8(sure_fg)






unknown =cv2.subtract(sure_bg,sure_fg)


ret3, markers = cv2.connectedComponents(sure_fg)
markers = markers+1
markers [unknown ==255] = 0
plt.imshow(markers , cmap = "jet")
plt.show()

markers = cv2.watershed(img1,markers)
img1[markers ==-1] = [255,0,255]
img2 = color.label2rgb(markers)

cv2.imshow("original image",img1)
cv2.imshow("Colored grains",img2)

#cv2.imshow("image", img)
cv2.imshow("Blur image", median)
cv2.imshow("real bg", sure_bg )
#cv2.imshow("eq_img", eq_img )
#cv2.imshow("bg e", erosion )
cv2.imshow("Theroshold image", thresh)
#cv2.imshow("Blur", img_Blur)
cv2.imshow("Theroshold image 2 ", thresh2)
#cv2.imshow("open image", opening )
cv2.imshow("Unknown Pixel", unknown)
# cv2.imshow("dist transform", dist_transform)
# cv2.imshow("bg", sure_bg)
cv2.imshow("sf", sure_fg  )
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
# from skimage.segmentation import clear_border
#
# opening = clear_border(opening)  # Remove edge touching grains
# # Check the total regions found before and after applying this.
#
#
# # Now we know that the regions at the center of cells is for sure cells
# # The region far away is background.
# # We need to extract sure regions. For that we can use erode.
# # But we have cells touching, so erode alone will not work.
# # To separate touching objects, the best approach would be distance transform and then thresholding.
#
# # let us start by identifying sure background area
# # dilating pixes a few times increases cell boundary to background.
# # This way whatever is remaining for sure will be background.
# # The area in between sure background and foreground is our ambiguous area.
# # Watershed should find this area for us.

#
# # Finding sure foreground area using distance transform and thresholding
# # intensities of the points inside the foreground regions are changed to
# # distance their respective distances from the closest 0 value (boundary).
# # https://www.tutorialspoint.com/opencv/opencv_distance_transformation.htm

#
# # Let us threshold the dist transform by 20% its max value.
# # print(dist_transform.max()) gives about 21.9

#
# # 0.2* max value seems to separate the cells well.
# # High value like 0.5 will not recognize some grain boundaries.
#
# # Unknown ambiguous region is nothing but bkground - foreground

#
# unknown = cv2.subtract(sure_bg, sure_fg)
#
# # Now we create a marker and label the regions inside.
# # For sure regions, both foreground and background will be labeled with positive numbers.
# # Unknown regions will be labeled 0.
# # For markers let us use ConnectedComponents.

#
# # One problem rightnow is that the entire background pixels is given value 0.
# # This means watershed considers this region as unknown.
# # So let us add 10 to all labels so that sure background is not 0, but 10
# markers = markers + 10
#
# # Now, mark the region of unknown with zero
# markers[unknown == 255] = 0
# # plt.imshow(markers, cmap='jet')   #Look at the 3 distinct regions.
#
# # Now we are ready for watershed filling.
# markers = cv2.watershed(img1, markers)
# # The boundary region will be marked -1
# # https://docs.opencv.org/3.3.1/d7/d1b/group__imgproc__misc.html#ga3267243e4d3f95165d55a618c65ac6e1
#
#
# # Let us color boundaries in yellow. OpenCv assigns boundaries to -1 after watershed.
# img1[markers == -1] = [0, 255, 255]
#
# img2 = color.label2rgb(markers, bg_label=0)
#
# cv2.imshow('Overlay on original image', img1)
# cv2.imshow('Colored Grains', img2)
# cv2.waitKey(0)
#
# # Now, time to extract properties of detected cells
# # regionprops function in skimage measure module calculates useful parameters for each object.
# regions = measure.regionprops(markers, intensity_image=img)
#
# # Can print various parameters for all objects
# # for prop in regions:
# #    print('Label: {} Area: {}'.format(prop.label, prop.area))
#
# # Best way is to output all properties to a csv file
# # Let us pick which ones we want to export.
#
# propList = ['Area',
#             'equivalent_diameter',  # Added... verify if it works
#             'orientation',  # Added, verify if it works. Angle btwn x-axis and major axis.
#             'MajorAxisLength',
#             'MinorAxisLength',
#             'Perimeter',
#             'MinIntensity',
#             'MeanIntensity',
#             'MaxIntensity']
#
# output_file = open('image_measurements.csv', 'w')
# output_file.write('Grain #' + "," + "," + ",".join(propList) + '\n')  # join strings in array by commas,
# # First cell to print grain number
# # Second cell blank as we will not print Label column
#
# grain_number = 1
# for region_props in regions:
#     output_file.write(str(grain_number) + ',')
#     # output cluster properties to the excel file
#     #    output_file.write(str(region_props['Label']))
#     for i, prop in enumerate(propList):
#         if (prop == 'Area'):
#             to_print = region_props[prop] * pixels_to_um ** 2  # Convert pixel square to um square
#         elif (prop == 'orientation'):
#             to_print = region_props[prop] * 57.2958  # Convert to degrees from radians
#         elif (prop.find('Intensity') < 0):  # Any prop without Intensity in its name
#             to_print = region_props[prop] * pixels_to_um
#         else:
#             to_print = region_props[prop]  # Reamining props, basically the ones with Intensity in its name
#         output_file.write(',' + str(to_print))
#     output_file.write('\n')
#     grain_number += 1
#
# output_file.close()