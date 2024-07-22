import numpy as np
import svgwrite
from PIL import Image
from image2parallel import draw_parallel
from matplotlib.colors import LinearSegmentedColormap
import requests
import os
import matplotlib.pyplot as plt
import random


if __name__ == '__main__':

	random.seed(42)
	np.random.seed(42)

	os.makedirs('output', exist_ok=True)

	url = 'https://upload.wikimedia.org/wikipedia/commons/1/1d/Katzepasstauf_%282009_photo%3B_cropped_2022%29_%28cropped%29.jpg'

	cat = np.asarray(Image.open(requests.get(url, stream=True).raw).resize((32, 32))).copy()
	
	
	drawing = svgwrite.Drawing('output/cat_rgb.svg', profile='tiny')
	draw_parallel(drawing, cat, s=50, use_rgb=True, margin=2)
	drawing.save()
	
	drawing = svgwrite.Drawing('output/cat_featuremap.svg', profile='tiny')
	draw_parallel(drawing, cat, s=50, use_rgb=False, cmap=plt.get_cmap('inferno'), margin=2)
	drawing.save()

	# filters

	cmap = LinearSegmentedColormap.from_list('mycmap', [(0, 'C0'), (0.5, 'white'), (1, 'C1')])
	cmap.set_under('lime')
	cmap.set_over('lime')
	cmap.set_bad('lime')

	pw_filter =  np.array([1, 1, 1]).reshape(1, 1, -1)
	drawing = svgwrite.Drawing('output/pointwise_filter.svg', profile='tiny')
	draw_parallel(drawing, pw_filter, s=50, use_rgb=False, cmap=cmap, margin=2, vmin=0, vmax=1)
	drawing.save()

	sobel_filter =  np.expand_dims(np.array([[1, 0, -1],
		   [2, 0, -2],
		   [1, 0, -1]]), axis=2).repeat(3, axis=2)
	drawing = svgwrite.Drawing('output/sobel_filter.svg', profile='tiny')
	draw_parallel(drawing, sobel_filter, s=50, use_rgb=False, cmap=cmap, margin=2)
	drawing.save()

	drawing = svgwrite.Drawing('output/sobel_kernel.svg', profile='tiny')
	draw_parallel(drawing, sobel_filter[:, :, 0], s=50, use_rgb=False, cmap=cmap, margin=2)
	drawing.save()

	blur_filter =  np.expand_dims(np.array([[1, 1, 1],
		   [1, 1, 1],
		   [1, 1, 1]]), axis=2).repeat(3, axis=2)
	drawing = svgwrite.Drawing('output/blur_filter.svg', profile='tiny')
	draw_parallel(drawing, blur_filter, s=50, use_rgb=False, cmap=cmap, vmin=0, vmax=0, margin=2)
	drawing.save()
	
	drawing = svgwrite.Drawing('output/blur_kernel.svg', profile='tiny')
	draw_parallel(drawing, blur_filter[:, :, 0], s=50, use_rgb=False, cmap=cmap, vmin=0, vmax=1, margin=2)
	drawing.save()

	rand_filter_1 = np.random.uniform(-1, 1, (3, 3, 3))

	_rf1 = rand_filter_1.copy()

	_rf1[0, 0, 0] = -100

	drawing = svgwrite.Drawing('output/random_filter_1.svg', profile='tiny')
	draw_parallel(drawing, _rf1, s=50, use_rgb=False, cmap=cmap, margin=2, vmin=-1, vmax=1)
	drawing.save()

	rand_filter_2 = np.random.uniform(-1, 1, (3, 3, 3))

	_rf2 = rand_filter_2.copy()
	_rf2[:, :, 0] = -100 

	drawing = svgwrite.Drawing('output/random_filter_2.svg', profile='tiny')
	draw_parallel(drawing, _rf2, s=50, use_rgb=False, cmap=cmap, margin=2, vmin=-1, vmax=1)
	drawing.save()

	# convolutions
	from scipy import signal
	
	def conv(x, w, depthwise=False):
		y = 0
		if not depthwise:
			for i in range(x.shape[2]):
				y += signal.convolve2d(x[:,:,i], w[:,:,i], boundary='fill', mode='same')
		else:
			y = []
			for i in range(x.shape[2]):
				y.append(signal.convolve2d(x[:,:,i], w[:,:,i], boundary='fill', mode='same'))
			y = np.stack(y, axis=2)
		return y
	
	y = conv(cat, sobel_filter, depthwise=False)
	
	drawing = svgwrite.Drawing('output/conv_cat_sobel.svg', profile='tiny')
	draw_parallel(drawing, y, s=50, use_rgb=False, cmap=plt.get_cmap('inferno'), margin=2)
	drawing.save()

	y = conv(cat, sobel_filter, depthwise=True)
	
	drawing = svgwrite.Drawing('output/conv_depthwise_cat_sobel.svg', profile='tiny')
	draw_parallel(drawing, y, s=50, use_rgb=False, cmap=plt.get_cmap('inferno'), margin=2)
	drawing.save()


	y = [
		conv(cat, rand_filter_1, depthwise=False),
		conv(cat, rand_filter_2, depthwise=False),
		conv(cat, blur_filter, depthwise=False),
		conv(cat, sobel_filter, depthwise=False)
	]
	y = np.stack(y, axis=2)

	drawing = svgwrite.Drawing('output/conv_cat_layer.svg', profile='tiny')
	draw_parallel(drawing, y, s=50, use_rgb=False, cmap=plt.get_cmap('inferno'), margin=2)
	drawing.save()
	

