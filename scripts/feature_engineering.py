# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:18:14 2019

@author: sebastien.delandtsheer@uni.lu

"""
def add_polynomials(df, degree = 2):
	from sklearn.preprocessing import PolynomialFeatures as pl
	import pandas as pd

	poly = pl(degree)
	dfPoly = poly.fit_transform(df)
	
	dfPoly = pd.DataFrame(data = dfPoly, index = df.index)
	
	dfPoly = pd.concat([df, dfPoly], axis = 1)

	return dfPoly

def categorize_data(df):
	
	import pandas as pd
	
	df_orig = df.copy()
	dfCat = pd.DataFrame()
	
	for col in df_orig.columns:
		tmp = pd.Categorical(df[col])
		dfDummies = pd.get_dummies(tmp, prefix = 'Sens=')
		dfCat = pd.concat([dfCat, dfDummies], axis = 1)
		
	return dfCat

