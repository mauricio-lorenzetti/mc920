import numpy as np
import cv2

### AUXILIARY FUNCTIONS AND VARIABLES
img_extension = ".png"
def isEven(a):
	return a%2 == 0

### Intensity
def intensity_tests(img_path):
	original_img = cv2.imread(img_path+img_extension, 0)

	### negative
	img = original_img.copy()

	for i in range (img.shape[0]):
		for j in range (img.shape[1]):
			img[i][j] = 255-img[i][j]

	cv2.imwrite(img_path+"_negative"+img_extension, img)

	### transform
	img = original_img.copy()

	for i in range (img.shape[0]):
		for j in range (img.shape[1]):
			img[i][j] = 100 + 100*img[i][j]/255

	cv2.imwrite(img_path+"_transform"+img_extension, img)

	### reversed even lines
	img = original_img.copy()

	for i in range (img.shape[0]):
		if isEven(i):
			for j in range (img.shape[1]):
				img[i][j] = original_img[i][img.shape[1]-j-1]

	cv2.imwrite(img_path+"_even"+img_extension, img)

	### mirrored
	img = original_img.copy()

	for i in range (img.shape[0]):
		if i > img.shape[0]/2:
			img[i] = original_img[img.shape[0]-i-1]
		else:
			img[i] = original_img[i]

	cv2.imwrite(img_path+"_mirror"+img_extension, img)

	### vertical mirror
	img = original_img.copy()

	for i in range (img.shape[0]):
		img[i] = original_img[img.shape[0]-i-1]

	cv2.imwrite(img_path+"_vertical"+img_extension, img)


### Brightness
def brightness_tests(img_path):
	original_img = cv2.imread(img_path+img_extension, 0)

	for gamma in [1.5, 2.5, 3.5]:

		img = original_img.copy()

		for i in range (img.shape[0]):
			for j in range (img.shape[1]):
				img[i][j] = ((original_img[i][j]/255.0) ** (1/gamma)) * 255

		cv2.imwrite(img_path+"_gamma_"+str(gamma)+img_extension, img)

### Bit planes
def bit_plane_extraction(img_path):
	original_img = cv2.imread(img_path+img_extension, 0)

	for bit_plane in range(8):

		img = original_img.copy()

		for i in range (img.shape[0]):
			for j in range (img.shape[1]):
				img[i][j] = 255 if format(original_img[i][j],"08b")[7-bit_plane] == "1" else 0

		cv2.imwrite(img_path+"_plane_"+str(bit_plane)+img_extension, img)


### Mosaic
def mosaic(img_path):
	original_img = cv2.imread(img_path+img_extension, 0)

	img = original_img.copy()

	original_shape = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
	final_shape = [[6,11,13,3],[8,16,1,9],[12,14,2,7],[4,15,10,5]]

	tile_width = int(img.shape[0]/4)
	tile_height = int(img.shape[1]/4)

	for index_original_line, original_line in enumerate(original_shape):
		for index_original_column, original_value in enumerate(original_line):
			for index_final_line, final_line in enumerate(final_shape):
				for index_final_column, final_value in enumerate(final_line):
					if original_value == final_value:
						i = 0
						j = 0
						start_i = tile_height*index_original_line
						start_j = tile_width*index_original_column
						for i_original in range(start_i, start_i + tile_height):
							for j_original in range(start_j, start_j + tile_width):
								img[tile_height*index_final_line+i_original-start_i][tile_width*index_final_column+j_original-start_j] = original_img[i_original][j_original]
								j+=1
							i+=1

	cv2.imwrite(img_path+"_mosaic"+img_extension, img)

### Combination
def combination(img_path_1, img_path_2):
	original_img_1 = cv2.imread(img_path_1+img_extension, 0)
	original_img_2 = cv2.imread(img_path_2+img_extension, 0)

	img = original_img_1.copy()

	for w in [[0.2,0.8],[0.5,0.5],[0.8,0.2]]:
		for i in range(img.shape[0]):
			for j in range(img.shape[1]):
				img[i][j] = int(np.average([original_img_1[i][j], original_img_2[i][j]], weights = w))
		cv2.imwrite(img_path_1+"_combination_"+str(w[0])+"_"+str(w[1])+img_extension, img)

### Filter
def apply_filters(img_path):
	original_img = cv2.imread(img_path+img_extension, 0)

	img = original_img.copy()

	#h1
	kernel = np.array([[0, 0, -1, 0, 0],
	                   [0, -1, 2, -1, 0],
	                   [-1, -2, 16, -2, -1],
	                   [0, -1, 2, -1, 0],
	                   [0, 0, -1, 0, 0]
	                   ])

	img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)

	cv2.imwrite(img_path+"_filter_h1"+img_extension, img)

	#h2
	kernel = np.array([[1, 4, 6, 4, 1],
	                   [4, 16, 24, 16, 4],
	                   [6, 24, 36, 24, 6],
	                   [4, 16, 24, 16, 4],
	                   [1, 4, 6, 4, 1]
	                   ]) * (1/256)

	img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)

	cv2.imwrite(img_path+"_filter_h2"+img_extension, img)

	#h3
	kernel = np.array([[-1, 0, 1],
	                   [-2, 0, 2],
	                   [-1, 0, 1]])

	img_3 = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)

	cv2.imwrite(img_path+"_filter_h3"+img_extension, img_3)

	#h4
	kernel = np.array([[-1, -2, -1],
	                   [0, 0, 0],
	                   [1, 2, 1]])

	img_4 = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)

	cv2.imwrite(img_path+"_filter_h4"+img_extension, img_4)

	#h3+h4
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			img[i][j] = (img_3[i][j] ** 2 + img_4[i][j] ** 2) ** (1/2)

	cv2.imwrite(img_path+"_filter_h3_h4_combined"+img_extension, img)

	#h5
	kernel = np.array([[-1, -1, -1],
	                   [-1, 8, -1],
	                   [-1, -1, -1]])

	img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)

	cv2.imwrite(img_path+"_filter_h5"+img_extension, img)

	#h6
	kernel = np.array([[1, 1, 1],
	                   [1, 1, 1],
	                   [1, 1, 1]]) * (1/9)

	img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)

	cv2.imwrite(img_path+"_filter_h6"+img_extension, img)

	#h7
	kernel = np.array([[-1, -1, 2],
	                   [-1, 2, -1],
	                   [2, -1, -1]])

	img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)

	cv2.imwrite(img_path+"_filter_h7"+img_extension, img)

	#h8
	kernel = np.array([[2, -1, -1],
	                   [-1, 2, -1],
	                   [-1, -1, 2]])

	img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)

	cv2.imwrite(img_path+"_filter_h8"+img_extension, img)

	#h9
	kernel = np.array([[-1, -1, -1, -1, -1],
	                   [-1, 2, 2, 2, -1],
	                   [-1, 2, 8, 2, -1],
	                   [-1, 2, 2, 2, -1],
	                   [-1, -1, -1, -1, -1]
	                   ]) * (1/8)

	img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)

	cv2.imwrite(img_path+"_filter_h10"+img_extension, img)

	#h10
	kernel = np.identity(9) * (1/9)

	img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)

	cv2.imwrite(img_path+"_filter_h9"+img_extension, img)

	#h11
	kernel = np.array([[-1, -1, 0],
	                   [-1, 0, 1],
	                   [0, 1, 1]])

	img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)

	cv2.imwrite(img_path+"_filter_h11"+img_extension, img)

images_path = "images/"
images = ["baboon", "butterfly", "city", "house", "seagull"]

for img in images:
	intensity_tests(images_path+img)
	brightness_tests(images_path+img)
	bit_plane_extraction(images_path+img)
	mosaic(images_path+img)
	combination(images_path+img, images_path+images[0])
	apply_filters(images_path+img)
