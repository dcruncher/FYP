from flask import Flask,render_template,request
from werkzeug.utils import secure_filename
import io,csv,pickle
import numpy as np
import pandas as pd
from zipfile import ZipFile 
import lightgbm as lgb
from sklearn.tree import DecisionTreeClassifier
import pickle
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
app = Flask(__name__)

df = pd.DataFrame()
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
      #df= prediction(df)

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

def prediction(test):
  
  # with open('xgb_pickle','rb') as f:
  #   model1=pickle.load(f)
  with open('dt_pickle','rb') as f:
    model2=pickle.load(f)
  with open('lightgbm_pickle','rb') as f:
    model3=pickle.load(f)

  #pred1=model1.predict(test)
  pred2=model2.predict(test)
  y_pred=model3.predict(test)

  #rounding the values
  y_pred=y_pred.round(0)
  #converting from float to integer
  pred3=y_pred.astype(int)

  #y_pred=pred1+pred2+pred3
  y_pred=pred2+pred3
  y_pred=pd.DataFrame(data=y_pred)
  y_pred[0][y_pred[0]>1]=1

  test["label"]=y_pred
  return test.iloc[:,:-1][test['label']==1]

@app.route('/stats')
def get_stats():
  gender_stat(df)
  return render_template('stats.html')
  #decide images and get image links
  #send image links to new html page through render_template
def gender_stat(temp):
  y = np.array([temp['gender'][temp['gender']==0].shape[0],temp['gender'][temp['gender']==1].shape[0]])
  mylabels = ["Female", "Male"]
  plt.pie(y,colors=['lightskyblue', 'lightcoral'],labels=mylabels,autopct='%1.1f%%', radius=0.2,center=(0.5,0.5),frame=False, pctdistance=0.5)
  plt.axis('equal')
  plt.legend(title='Gender',bbox_to_anchor=(1,0.5),labels=mylabels,loc="center right",bbox_transform=plt.gcf().transFigure)
  plt.savefig('static\gender_stat.jpg')
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
if(__name__)=='__main__':
    app.run(debug=True)