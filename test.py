import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

plt.figure()

d = pd.concat([pr, df], axis=1)

# Create the scatter plot
sns.scatterplot(x='pred2_RFC', y='PD-0325901_ActArea', data=d)
plt.xlabel('Prediction')
plt.ylabel('Sensitivity (ActArea)')
plt.show()
# Show the plot
plt.savefig('PD0325901')

x = d['pred2_RFC']
y = d['PD-0325901_ActArea']
model = LinearRegression().fit(x.values.reshape(-1,1), y)
r_sq = model.score(x.values.reshape(-1,1), y)



d['r'] = np.where(((d['pred2_RFC'] > 0.5) & (d['PD-0325901_ActArea'] > 2)) | ((d['pred2_RFC'] < 0.5) & (d['PD-0325901_ActArea'] < 2)), 1, 0)

d = d.dropna()

dn = d[d['pred2_RFC'] < 0.5]
dp = d[d['pred2_RFC'] > 0.5]

ppv = dp['r'].mean()
npv = dn['r'].mean()