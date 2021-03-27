from flask import Flask,render_template,request
# from flask_cache import Cache
from werkzeug.utils import secure_filename
import io,csv,pickle
import numpy as np
import operator
import pandas as pd
from zipfile import ZipFile 
import lightgbm as lgb
from sklearn.tree import DecisionTreeClassifier
import pickle
from datetime import date
from sklearn import preprocessing
#from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import sklearn
from pickle import load
from datetime import date
from sklearn import preprocessing
# cache = Cache(config={'CACHE_TYPE': 'null'})
app = Flask(__name__)
# cache = Cache(app,config={'CACHE_TYPE': 'simple'})
# app.config['TESTING']= True
df = pd.DataFrame()
total=0
fraud=0
n_fraud=0
@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response
@app.route('/')
def hello():
  return render_template('home_page.html')
#check for zip extention, correct columns and column count
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      file = f.filename
      name =file
      global df
      df= pd.read_csv(name)
      df = df.iloc[:,0:-1]
      refer_cols=['trans_num','category','amt', 'gender','state', 'zip', 'lat', 'long', 'city_pop', 'job','unix_time', 'merch_lat', 'merch_long','trans_date','age']
      data_cols = df.columns
      print(data_cols)
      if not len(data_cols)==len(refer_cols):
        return '<script>alert("columns not matched. please choose file again")</script>'+ render_template('home_page.html')
      elif not all(refer_cols==data_cols):
        return '<script>alert("columns not matched. please choose file again")</script>'+ render_template('home_page.html')
      
      # trans_num = prediction(df)
      # trans_num=pd.DataFrame(data=trans_num)
      # df = pd.merge(trans_num,df)
      df = prediction(data_processing(df))

      ##code to create new dataframe
      f=open("templates/result.html","w")
      f.write('<table id="example" style="width:100%" class="table table-striped table-bordered">')
      f.write("<thead><tr><th>trans_num</th><th>Category</th><th>Amount</th><th>Gender</th><th>State</th><th>Zip</th><th>Latitude</th><th>Longitude</th><th>City Population</th><th>Job</th><th>Unix Time</th><th>Merchant Latitude</th><th>Merchant Longitude</th><th>Transaction Date</th><th>Age</th></tr></thead>")
      # with open(name, 'r') as fi:
      #   Reader = csv.reader(fi)
      #   Data = list(Reader)
      #   for data in Data :
      f.write("<tbody>")

      rows = len(df.axes[0])
      cols = len(df.axes[1])
      for i in range(rows):
        f.write("<tr>")
        for j in range(cols):
          f.write("<td>"+ str(df.iloc[i,j])+"</td>")
        f.write("</tr>")
      f.write("</tbody></table>")
      f.close()
      print(df.shape)
      return render_template('result_assist_start.html')+render_template('result.html')+render_template('result_assist_end.html')
      # +  render_template('home_page.html')
def data_processing(data):
  df=data.iloc[:,:].copy()
  l=load(open('le.pkl', 'rb'))
  l1=load(open('le1.pkl', 'rb'))
  l2=load(open('le2.pkl', 'rb'))
  l3=load(open('le3.pkl', 'rb'))
  sc=load(open('scaler.pkl', 'rb'))
  df["category"]=l.transform(df["category"])
  df["state"]=l1.transform(df["state"])
  df["job"]=l2.transform(df["job"])
  df["gender"]=l3.transform(df["gender"])

  df.loc[:,["amt","lat","long","city_pop","unix_time","merch_lat","merch_long","trans_date","age"]]=sc.transform(df.loc[:,["amt","lat","long","city_pop","unix_time","merch_lat","merch_long","trans_date","age"]])

  
  # df.to_csv("/content/drive/MyDrive/Final Year Project/new_test.csv")

  test=df.copy()
  return test


def prediction(test):
  
  # with open('xgb_pickle','rb') as f:
    # model1=pickle.load(f)
  with open('dt_pickle','rb') as f:
    model2=pickle.load(f)
  with open('lightgbm_pickle','rb') as f:
    model3=pickle.load(f)
   
  # pred1=model1.predict(test.iloc[:,1:])
  pred2=model2.predict(test.iloc[:,1:])
  y_pred=model3.predict(test.iloc[:,1:])

  #rounding the values
  y_pred=y_pred.round(0)
  #converting from float to integer
  pred3=y_pred.astype(int)
  global total,fraud,n_fraud
  # y_pred=pred1+pred2+pred3
  y_pred=pred2+pred3
  y_pred=pd.DataFrame(data=y_pred)
  y_pred[0][y_pred[0]>1]=1

  test["label"]=y_pred
  total = test.shape[0]
  isfraud=test.loc[:,"trans_num"][test['label']==1]
  isfraud=pd.DataFrame(data=isfraud)
  fraud = isfraud.shape[0]
  n_fraud= total- fraud
  data=pd.read_csv("fetest1.csv").loc[:,['trans_num','category','amt', 'gender','state', 'zip', 'lat', 'long', 'city_pop', 'job','unix_time', 'merch_lat', 'merch_long','trans_date','age']]
  
  result = pd.merge(isfraud, data, on="trans_num")
  return result

@app.route('/stats')
def get_stats():
  gender_stat(df)
  age_stat(df)
  amt_stat(df)
  cat_stat(df)
  state_stat(df)
  over_stat()
  return render_template('stats.html')
  #decide images and get image links
  #send image links to new html page through render_template
def gender_stat(temp):
  y = np.array([temp['gender'][temp['gender']=='F'].shape[0],temp['gender'][temp['gender']=='M'].shape[0]])
  mylabels = ["Female", "Male"]
  plt.pie(y,colors=['lightskyblue', 'lightcoral'],labels=mylabels,autopct='%1.1f%%', radius=0.2,center=(0.5,0.5),frame=False, pctdistance=0.5)
  plt.axis('equal')
  plt.legend(title='Gender',bbox_to_anchor=(1,0.5),labels=mylabels,loc="center right",bbox_transform=plt.gcf().transFigure)
  plt.savefig('static/gender_stat.jpg')
  plt.clf()
  return   
def age_stat(temp):
  age=[]
  age.append(temp['age'][temp['age']<20].shape[0])
  age.append(temp['age'][(temp['age']>=20)&(temp['age']<40)].shape[0])
  age.append(temp['age'][(temp['age']>=40)&(temp['age']<60)].shape[0])
  age.append(temp['age'][(temp['age']>=60)].shape[0])

  colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']

  y = np.array(age)
  mylabels = ["<20", "20-39","40-59",">=60"]

  plt.pie(y, labels = mylabels, startangle = 90,explode=[0.2,0,0,0],colors=colors,autopct='%1.1f%%', radius=2,center=(0.5,0.5),frame=False, pctdistance=0.5)
  plt.legend(title='Age',bbox_to_anchor=(1,0.5),loc="center right",bbox_transform=plt.gcf().transFigure)
  plt.tight_layout()
  plt.savefig('static/age_stat.jpg')
  plt.clf()
def amt_stat(temp):
  y = [temp.iloc[:,:][temp['amt']<200].shape[0],temp.iloc[:,:][(temp['amt']>=200) &(temp['amt']<400)].shape[0],temp.iloc[:,:][(temp['amt']>=400) &(temp['amt']<600)].shape[0],temp.iloc[:,:][(temp['amt']>=600) &(temp['amt']<800)].shape[0],temp.iloc[:,:][(temp['amt']>=800) &(temp['amt']<1000)].shape[0],temp.iloc[:,:][(temp['amt']>=1000)].shape[0]]
  x = [i for i in range(1,7)]
  label=['<200','200-400','400-600','600-800','800-1000','>1000']


  for i in range(0,6):
    plt.annotate(label[i],(x[i],y[i]))

  plt.plot(x, y,marker='o',label="Amount",color='red')
  plt.ylabel("No of Instances")
  plt.xlabel("Category Number")

  plt.legend()
  plt.savefig('static/amount_stat.jpg')
  plt.clf()

def cat_stat(temp):
  cat={}
  for i in list(temp["category"].unique()):
    cat[i]=temp["category"][temp["category"]==i].shape[0]
  

  sorted_tuples = sorted(cat.items(), key=operator.itemgetter(1),reverse=True)
  sorted_dict = {k: v for k, v in sorted_tuples}

  x=list(sorted_dict.keys())[0:6]
  y = list(sorted_dict.values())[0:6] 
  fig = plt.figure(figsize = (10, 5))
  
  # creating the bar plot
  plt.bar(x, y, color ='purple',
          width = 0.4)
  
  plt.xlabel("Category Name")
  plt.ylabel("No. of Instances")
  plt.savefig('static/category_stat.jpg')
  plt.clf()
def state_stat(temp):
  state={}
  for i in list(temp["job"].unique()):
    state[i]=temp['job'][temp['job']==i].shape[0]

  sorted_tuples = sorted(state.items(), key=operator.itemgetter(1),reverse=True)
  sorted_dict1 = {k: v for k, v in sorted_tuples}
  y = list(sorted_dict1.values())[0:6] 
  x = [i for i in range(1,7)]
  label=list(sorted_dict1.keys())[0:6]
  # plt.ylim([30000,140000])
  plt.ylabel("No of Instances")
  for i in range(0,6):
    plt.annotate(label[i],(x[i],y[i]))
  plt.scatter(x, y)
  plt.savefig('static/state_stat.jpg')
  plt.clf()
def over_stat():
  fp = (fraud/total)*100
  np = (n_fraud/total)*100
  y = [fp,np]
  x=[1,2]
  label= ['Fraud','Not Fraud']
  plt.ylabel('In Percentage')
  for i in range(0,2):
    plt.annotate(label[i],(x[i],y[i]))
  plt.bar(x,y,width=0.4)
  plt.savefig('static/overstat.jpg')
  plt.clf() 


  
if(__name__)=='__main__':
    app.run(debug=True)