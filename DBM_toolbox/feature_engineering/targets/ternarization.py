# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 10:58:05 2020

@author: sebde
"""
# import numpy as np
import pandas as pd

from sklearn.preprocessing import binarize

# from scipy.signal import find_peaks
# from scipy.stats.kde import gaussian_kde
# from astropy import modeling

def get_drug_response(df, thresholdR = 0, thresholdS = 0, axis = 'columns'):
	"""replaces the data with indication of sensitivity based on quantiles:
		-1 = Resistant
		0 = Intermediate
		1 = Sensitive
		values in df indicate response to drug (like Act Area)
		"""
	if isinstance(thresholdS, int):
		dfSens = binarize(df, threshold = thresholdS) #Resistants will get 0, while Intermediates and Sensitives get 1
	else:
		dfSens = binarize(df.sub(thresholdS, axis=axis))
	if isinstance(thresholdR, int):
		dfNotR = binarize(df, threshold = thresholdR) #Sensitives get 1 more, others get 0
	else:
		dfNotR = binarize(df.sub(thresholdR, axis=axis))
	dfAll = dfSens + dfNotR -1
	dfAll = pd.DataFrame(data = dfAll, index=df.index, columns=df.columns)
	return dfAll



def ternarize_targets_density(df, ):
	"""replaces the data with indication of sensitivity based on density:
		-1 = Resistant
		0 = Intermediate
		1 = Sensitive
		values in df indicate response to drug (like Act Area)
		"""
	### TODO: Maria
	
	
# def process_RNA_expressions(df):
# 	"""performs discretization of the RNA expression levels into three levels
# 	old function not used at the moment """
# 	
# 	dftrans = df
# 	dftrans_dropped=dftrans #.drop(columns=['Name', 'Description'])
# 	dftrans_dropped.fillna(0, inplace=True) # fill empty by 0
# 	dftrans_log = np.log2(dftrans_dropped) # log transform
# 	# replace -inf by -1000
# 	dftrans_log.replace(-np.inf, -1000, inplace=True)
# 	df_z_trans=dftrans.copy()
# 	di_dis=dftrans.copy()

# 	siz=dftrans_log.shape[1]
# 	for i in range(0,siz):
# 		print(i)
# 		df_sample=dftrans_log.iloc[:,i] # consider each sample individually
# 		df_sample2=df_sample[df_sample>-1000] # remove values that were equal to zero in the original data
# 		my_pdf = gaussian_kde(df_sample2)
# 		x = np.linspace(min(df_sample2),max(df_sample2),100)

# 		# find the peaks in the data
# 		indices = find_peaks(my_pdf(x))
# 		type(indices)
# 		indices2=indices[0]
# 		my_pdf(x[indices2])
# 		
# 		# find the maximum
# 		a=my_pdf(x[indices2])==max(my_pdf(x[indices2]))
# 		b=my_pdf(x[indices2])!=max(my_pdf(x[indices2]))
# 		max_peak_y= my_pdf(x[indices2[a]])
# 		max_peak_x= x[indices2[a]]
# 		indices_max=indices2[a]
# 		# find the remaining peaks
# 		remaining_peaks_y=my_pdf(x[indices2[b]])
# 		remaining_peaks_x=x[indices2[b]]
# 		indices_remaining=indices2[b]
# 		remaining_peaks_x=remaining_peaks_x[remaining_peaks_y > 0.01 * max_peak_y]
# 		indices_remaining=indices2[b]
# 		indices_remaining=indices_remaining[remaining_peaks_y > 0.01 * max_peak_y]
# 		remaining_peaks_y=remaining_peaks_y[remaining_peaks_y > 0.01 * max_peak_y]
# 		if len(indices_remaining)==1:
# 			inds=max(indices_max, indices_remaining)
# 		elif len(indices_remaining)>1:
# 			next_to_match=(abs(indices_remaining-indices_max))
# 			indices_remaining=indices_remaining[next_to_match==max(next_to_match)]
# 			inds=max(indices_max, indices_remaining)
# 		a=my_pdf(x[inds[0] :])
# 		b=a[::-1]
# 		c=np.concatenate((b,a), axis=0)
# 		c=c[-100:]
# 		x2=x[-len(c):]
# 		model = modeling.models.Gaussian1D()
# 		fitter = modeling.fitting.LevMarLSQFitter()
# 		fitted_model = fitter(model, x2, c)
# 		m=fitted_model.mean
# 		s=fitted_model.stddev
# 		zFRMA=(df_sample2-m)/s
# 		my_pdf2 = gaussian_kde(zFRMA)
# 		y=min(indices_remaining,indices_max*2)
# 		noise=my_pdf(x)-fitted_model(x)
# 		noise=noise[0:y[0]-1]
# 		x3=x[0:y[0]-1]
# 		model = modeling.models.Gaussian1D(0.1,x[y], 5)
# 		fitted_model2 = fitter(model, x3,noise)
# 		m2=fitted_model2.mean
# 		s2=fitted_model2.stddev
# 		m2=(m2-m)/s2
# 		m2=max(m2[0],-3)
# 		Dis_data=zFRMA.copy()
# 		Dis_data[zFRMA>0]=1
# 		Dis_data[zFRMA<m2]=-1
# 		Dis_data[(zFRMA>m2) & (zFRMA<0)]=0
# 		ind = np.where(df_sample > -1000)
# 		df_z_trans.iloc[ind[0],i] = zFRMA
# 		df_z_trans.copy()
# 		di_dis.iloc[ind[0],i] =Dis_data
# 	return di_dis.transpose()





