# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 11:15:55 2020

@author: sebde
"""

from matplotlib import pyplot as plt

def plot_importances(coef, names):
	imp = coef
	imp,names = zip(*sorted(zip(imp,names)))
	plt.barh(range(len(names)), imp, align='center')
	plt.yticks(range(len(names)), names)
	plt.show()