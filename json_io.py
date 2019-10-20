import sys
from flask import Flask, flash, redirect, render_template,request, url_for
import numpy as np
import pandas as pd
#from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
import seaborn as sns
import mpl_toolkits
#%matplotlib inline

#import pprint as ppfrom Flask

app = Flask(__name__)

@app.route('/')
def output():
	return render_template("index.html")

@app.route("/result", methods=['GET','POST'])
def result():
	country = request.form['Country']
	brand = request.form['Brand']
	totalSize = request.form.get('TotalPackageSize', type=float)
	unitSize = request.form.get('UnitPackageSize', type=float)
	result = colgate(country,brand,totalSize,unitSize)
	print(result[0])
	return render_template("result.html", result = result)

def colgate(country,brand,totalSize,unitSize):
	data = pd.read_csv("hack_ru.csv")
	data.drop(data[data['company'] != brand].index,inplace = True)
	print(data.head())
	data.drop(data[data['country'] != country].index,inplace = True)

	data.drop(data[data['unit_pack_size_ml_g'] >300].index, inplace = True) 
	data.drop(data[data['price_per_100g_ml_dollars'] > 20].index, inplace = True) 
	counte=data['company'].count()
	if(counte<10):
	    return("bad")

	from sklearn.ensemble import GradientBoostingRegressor
	reg1 = GradientBoostingRegressor(n_estimators=12000, max_depth=4)
	reg2 = GradientBoostingRegressor(n_estimators=12000, max_depth=4)
	labels = data['price_per_100g_ml_dollars']
	from sklearn.preprocessing import LabelEncoder
	train1 = data.drop(['country','company','ingredients','Unnamed: 0','total_pack_size_ml_g','price_per_100g_ml_dollars'], axis=1)
	train2 = data.drop(['country','company','ingredients','Unnamed: 0','unit_pack_size_ml_g','price_per_100g_ml_dollars'], axis=1)
	from sklearn.model_selection import train_test_split
	x_train1, x_test1, y_train1, y_test1 = train_test_split(train1, labels, test_size=0.1, random_state=2)
	x_train2, x_test2, y_train2, y_test2 = train_test_split(train2, labels, test_size=0.1, random_state=2)
	reg1.fit(x_train1, y_train1)
	print(reg1.score(x_test1,y_test1))
	reg2.fit(x_train2, y_train2)
	print(reg2.score(x_test2,y_test2))
	if(totalSize!=None and unitSize!=None):
		if(reg2.score(x_test2,y_test2)>reg1.score(x_test1,y_test1)):
			totalSize2=np.array(totalSize)
			totalSize3=totalSize2.reshape(1,-1)
			y_pred2=reg2.predict(totalSize2)
			return(y_pred2)
		elif(reg2.score(x_test2,y_test2)<reg1.score(x_test1,y_test1)):
			unitSize2=np.array(unitSize)
			unitSize3=unitSize2.reshape(1,-1)
			y_pred1=reg1.predict(unitSize3)
			return(y_pred1)
		elif(totalSize!=None and unitSize==None):
			totalSize2=np.array(totalSize)
			totalSize3=unitSize2.reshape(1,-1)
			y_pred2=reg2.predict(totalSize2)
			return(y_pred2)
	elif (unitSize!=None and totalSize==None):
		unitSize2=np.array(unitSize)
		unitSize3=unitSize2.reshape(1,-1)
		y_pred1=reg1.predict(unitSize3)
		return(y_pred1)
	return 0

if __name__== '__main__':
	app.run(debug=True, use_reloader=True, port=8080)
