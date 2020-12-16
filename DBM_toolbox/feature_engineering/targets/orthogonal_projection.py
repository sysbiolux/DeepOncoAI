# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 10:03:18 2020

@author: sebde
"""
import numpy as np
from sklearn.linear_model import LinearRegression


def orthogonal_projection(x, y):
	'''
	Projects points on their best regression line and computes
	their scaled distances

	Parameters
	----------
	x : 1-D Numpy array
		x-axis coordinates.
	y : 1-D Numpy array
		y-axis coordinates.

	Returns
	-------
	Returns a tuple d, [a, b] where d is the scaled distances (d of min(x) = 0)
	and a and b are the slope and intercept of the least squares regression line.
	'''
	
	x = x.reshape(-1,1)
	y = y.reshape(-1,1)
	model = LinearRegression().fit(x, y)
	a = model.coef_
	b = model.intercept_
	xp = (x + a*y - a*b) / (a**2 +1)
	yp = a*xp + b
	minx = min(xp)
	miny = a*minx + b
	d = np.sqrt((xp-(minx))**2 + (yp-miny)**2)
	return d, [a, b]