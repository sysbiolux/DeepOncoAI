# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:18:14 2019

@author: sebastien.delandtsheer@uni.lu

"""
def add_polynomials(df, degree = 2):
	from sklearn.preprocessing import PolynomialFeatures as pl

	poly = pl(degree)
	dfPoly = poly.fit_transform(df)

	return dfPoly

def categorize_data(df):
	from sklearn.preprocessing import OneHotEncoder as ohe
	enc = ohe(handle_unknown='ignore')
	dfCat = enc.fit_transform(df)

	return dfCat

