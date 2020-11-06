from flask import Flask, render_template, request, flash,jsonify,redirect,url_for
from flask_bootstrap import Bootstrap

# Weather Prediction
from weather import Weather

# Image Classifier
import os
from pest import Pest
import numpy as np
from keras.preprocessing import image
from tensorflow.python.keras import backend as K
import tensorflow as tf
import pickle
from test import Predict
from keras.models import Sequential

# Market Stats
from market_stat import Market

# Crop Prediction
from crop_predict import Crop_Predict

# Fertilizer Info
import pandas as pd

# Firebase
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from test import Predict
from firebase_admin import auth

cred = credentials.Certificate('key.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# Firebase end Here

# Login
from login import Login

# Admin Login
from admin_login import Login_Admin

# Kisan Center Login
from kisan_center_login import Login_Kisan

# Twilo Message
from twilio.rest import Client

# Weather Forcast 15 Days
import requests
import bs4
import xml.etree.ElementTree as ET

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from csv import writer

def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in
        csv_writer.writerow(list_of_elem)

# Pest Detection
from tensorflow import keras
from skimage import io
from gevent.pywsgi import WSGIServer
from werkzeug.utils import secure_filename
model =tf.keras.models.load_model('PlantDNet.h5',compile=False)
print('Model loaded. Check http://127.0.0.1:5000/')

def model_predict(img_path, model):
    img = image.load_img(img_path, grayscale=False, target_size=(64, 64))
    show_img = image.load_img(img_path, grayscale=False, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = np.array(x, 'float32')
    x /= 255
    preds = model.predict(x)
    return preds

app = Flask(__name__)
Bootstrap(app)

app.config['SECRET_KEY'] = 'e53b7406a43e2fd9ec89553019420927'


@app.route('/')
def main():
    # return render_template('index.html')
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload_detection():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        print(preds[0])

        # x = x.reshape([64, 64]);
        disease_class = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight',
                         'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight',
                         'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
                         'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
                         'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']
        a = preds[0]
        ind=np.argmax(a)
        print('Prediction:', disease_class[ind])
        result=disease_class[ind]
        return result
    return None

@app.route('/weather',methods=['POST','GET'])
def weather():
    weatherModel = Weather()
    if request.method == 'POST':
        city_name = request.form['city']
        if len(city_name) == 0:
            return render_template('weather_pred.html',error=1)
        try:
            daily = request.form['daily']
            print(daily)
            valid = weatherModel.update(city_name)
            if valid == 'noData':
                return render_template('weather_pred.html',error=1)

            weather_data = weatherModel.display()
            # print()
            invalidZip = False
            results = {"zipcode":city_name,"invalidZip":invalidZip, "weather":weather_data}

            return render_template('weather.html',results=results)
        except:
            day_15 = request.form['15days']
            print(day_15)
            city_name = city_name.lower()
            print(city_name)
            res = requests.get('https://www.timeanddate.com/weather/india/'+city_name+'/ext')
            # data = bs4.BeautifulSoup(res.text,'lxml')
            data = bs4.BeautifulSoup(res.text,'lxml')

            temp=data.find_all("table",{"class":"zebra tb-wt fw va-m tb-hover"})


            #temp = data.find_all(name='table', attrs={'id':'wt-ext'})
            #temp = data.find_all('tr','c1')

            # type(temp)
            # temp
            lt = []
            for i,items in enumerate(temp):
                for i,row in enumerate(items.find_all("tr")):
                    dt = {}

                    try:
    #                  print(i , row.find_all("td",{"class":""})[0].text)
                       dt['day'] = row.find_all("th",{"class":""})[-1].text
                    except:
                       dt['day'] = np.nan

                    try:
    #                  print(i , row.find_all("td",{"class":""})[0].text)
                       dt['temp'] = row.find_all("td",{"class":""})[0].text
                    except:
                       dt['temp'] = np.nan

                    try:
    #                  print(i , row.find_all("td",{"class":""})[0].text)
                       dt['weather'] = row.find_all("td",{"class":"small"})[0].text
                    except:
                       dt['weather'] = np.nan

                    try:
    #                  print(i , row.find_all("td",{"class":""})[0].text)
                       dt['temp_max'] = row.find_all("td",{"class":"sep"})[0].text
                    except:
                       dt['temp_max'] = np.nan

                    try:
    #                  print(i , row.find_all("td",{"class":""})[0].text)
                       dt['wind_speed'] = row.find_all("td",{"class":""})[1].text
                    except:
                       dt['wind_speed'] = np.nan

                    try:
    #                  print(i , row.find_all("td",{"class":""})[0].text)
                       dt['max_humidity'] = row.find_all("td",{"class":""})[4].text
                    except:
                       dt['max_humidity'] = np.nan

                    try:
    #                  print(i , row.find_all("td",{"class":""})[0].text)
                       dt['min_humidity'] = row.find_all("td",{"class":""})[3].text
                    except:
                       dt['min_humidity'] = np.nan

                    try:
    #                  print(i , row.find_all("td",{"class":""})[0].text)
                       dt['sun_rise'] = row.find_all("td",{"class":""})[5].text
                    except:
                       dt['sun_rise'] = np.nan

                    try:
    #                  print(i , row.find_all("td",{"class":""})[0].text)
                       dt['sun_set'] = row.find_all("td",{"class":""})[6].text
                    except:
                       dt['sun_set'] = np.nan
                    
                    lt.append(dt)

            return render_template('weather_15_days.html',result=lt,result_len = len(lt))
       
    
    return render_template('weather_pred.html',error=0)



@app.route("/upload", methods=['POST','GET'])
# def upload():

#     if request.method == 'POST': 
#         pest = Pest()
#         arrary_image,img = pest.Upload()
#         print(arrary_image)

#         if arrary_image == 'noData':
#             return render_template('pest.html',display=1)
        
#         else:
        
#             model_file = pickle.load(open("model.pkl","rb"))
#             # model._make_predict_function()
#             global graph
#             p = Predict()
#             # graph = tf.get_default_graph()
#             graph = tf.compat.v1.get_default_graph()
#             with graph.as_default():
#                 predict = model_file.predict(arrary_image)
            
#             label_binarizer = pickle.load(open("label_transform.pkl",'rb'))
#             result = label_binarizer.inverse_transform(predict)[0]    
#             print(result)
#             x = p.predicts()
#             result = label_binarizer.inverse_transform(x)[0]
#             print(result)
#             K.clear_session()

#             if result == 'Pepper__bell___healthy' or result == 'Tomato_healthy' or result == 'Potato___healthy':

#                 return render_template('pest_predict.html',result=result,image_name=img)
            
#             else :
                
#                 doc_ref = db.collection(u'pest').document(u''+result)

#                 try:
#                     doc = doc_ref.get()
#                     # print(u'Document data: {}'.format(doc.to_dict()))
#                     doc = doc.to_dict()
#                     print(len(doc))
#                 except google.cloud.exceptions.NotFound:
#                     print(u'No such document!')
            
#                 # result = model.predict(test_image)
#                 # result =  model._make_predict_function(test_image)
#             return render_template('pest_predict.html',result=result,image_name=img,data=doc)
    
#     return render_template('pest.html',display=0)

def upload():
    # return render_template('index.html')
    return render_template('pest_detection.html')



  
@app.route('/market',methods=['POST','GET'])
def market():

    model = Market()
    states,crops = model.State_Crop()
    if request.method == 'POST':
        state = request.form['state']
        crop = request.form['crop']
        lt = model.predict_data(state,crop)

        return render_template('market.html',result=lt,result_len =len(lt),display=True,states=states,crops=crops)

    return render_template('market.html',states=states,crops=crops)   
    

@app.route('/crop',methods=['GET','POST'])
def crop():
# ['Fava beans (Papdi - Val)', 'pigeon peas(Toor Dal)', 'Mung beans', 'Rapeseed (Mohri)', 'Cumin seeds', 'Cauliflower']

    model = Crop_Predict()
    if request.method == 'POST':
        crop_name = model.crop()
        existing = request.form['existing']
        # print(existing)
        if crop_name == 'noData':
            return render_template('crop_prediction.html',error=1)

        dict_family = {'Rice':'Grasses', 'Jowar(Sorghum)':'Grasses', 'Barley(JAV)':'Grasses', 'Maize':'Grasses','Ragi( naachnnii)':'Grasses', 'Chickpeas(Channa)':'Legumes', 'French Beans(Farasbi)':'Legumes','Fava beans (Papdi - Val)':'Legumes', 'Lima beans(Pavta)':'Legumes','Cluster Beans(Gavar)':'Legumes', 'Soyabean':'Legumes', 'Black eyed beans( chawli)':'Legumes','Kidney beans':'Legumes', 'pigeon peas(Toor Dal)':'Legumes', 'Moth bean(Matki)':'Legumes','Mung beans':'Legumes', 'Green Peas':'Legumes', 'Horse Gram(kulthi)':'Legumes', 'Black Gram':'Legumes','Rapeseed (Mohri)':'Mustards', 'Coriander seeds':'Umbellifers', 'Mustard seeds':'Mustards','sesame seed':'Sesame', 'Cumin seeds':'Umbellifers', 'Lentils(Masoor Dal)':'Legumes', 'Brinjal':'Nightshade','Beetroot':'Roots', 'Bitter Gourd':'Cucurbits', 'Bottle Gourd':'Cucurbits', 'Capsicum':'Nightshade', 'Cabbage':'Mustards','Carrot':'Umbellifers', 'Cauliflower':'Mustards', 'Cucumber':'Cucurbits', 'Coriander leaves':'Umbellifers','Curry leaves':'Rutaceae', 'Drumstick – moringa':'Moringaceae', 'Chili':'Nightshade', 'Lady Finger':'Mallows','Mushroom':'Agaricaceae', 'Onion':'Onion', 'Potato':'Nightshade', 'Pumpkin':'Cucurbits', 'Radish':'Mustards', 'Olive':'Olives','Sweet Potato':'Morning-glory', 'Fenugreek Leaf(methi)':'Pea', 'Spinach':'Goosefoot', 'Ridgegourd':'Cucurbits','Gooseberry(Amla)':'Phyllanthaceae', 'Jambun(Syzygium cumini)':'Myrtle','Ziziphus mauritiana(Bor)':'Buckthorns', 'Garcinia indica(kokam)':'Guttiferae', 'Tamarind':'Legumes','Tapioca(Suran)':'Euphorbia', 'Garlic':'Onion', 'Lemon':'Rutaceae', 'Tomato':'Nightshade', 'Ash Gourd':'Cucurbits','Pineapple':'Bromeliads', 'Pomegranate':'Punicaceae', 'Banana':'Musaceae', 'Mango':'Cashews', 'Grapes':'Grape','Jackfruit':'Mulberry', 'Guava':'Myrtle', 'Water Melon':'Cucurbits', 'Musk Melon':'Cucurbits', 'Apricot':'Rose','Apple':'Rose', 'Chickoo':'Sapotaceae', 'Custard apple':'Annonaceae', 'Dates':'Palm', 'Figs':'Mulberry', 'Orange':'Rutaceae','Papaya':'Caricaceae', 'Aniseed':'Umbellifers', 'Asafoetida':'Umbellifers', 'Bay Leaf':'Laurels', 'Black Pepper':'Piperaceae','Cardamom':'Ginger', 'Cinnamon':'Laurels', 'Cloves':'Myrtle', 'Jaiphal(Nutmeg)':'Myristicaceae', 'Ginger':'Ginger','Turmeric':'Ginger', 'Cashewnuts':'Cashews', 'Raisins':'Grapevine', 'Coconut':'Palm', 'Almond Nut':'Rose','Arecanut':'Palm', 'Pistachio Nut':'Cashews', 'Lemon Grass':'Grasses', 'Cotton':'Mallows', 'Jute':'Mallows','Coffee':'Rubiales', 'Sunflower':'Sunflower'}
    
        Grasses = ['Rice', 'Jowar(Sorghum)', 'Barley(JAV)', 'Maize', 'Ragi( naachnnii)', 'Lemon Grass']
        Cucurbits = ['Bitter Gourd', 'Bottle Gourd', 'Cucumber', 'Pumpkin', 'Ridgegourd', 'Ash Gourd', 'Water Melon', 'Musk Melon']
        Legumes = ['Chickpeas(Channa)', 'French Beans(Farasbi)', 'Fava beans (Papdi - Val)', 'Lima beans(Pavta)', 'Cluster Beans(Gavar)', 'Soyabean', 'Black eyed beans( chawli)', 'Kidney beans', 'pigeon peas(Toor Dal)', 'Moth bean(Matki)', 'Mung beans', 'Green Peas', 'Horse Gram(kulthi)', 'Black Gram', 'Lentils(Masoor Dal)', 'Tamarind']
        Umbellifers = ['Coriander seeds', 'Cumin seeds', 'Carrot', 'Coriander leaves', 'Aniseed', 'Asafoetida']
        Mustards =  ['Rapeseed (Mohri)', 'Mustard seeds', 'Cabbage', 'Cauliflower', 'Radish']
        Nightshade =  ['Brinjal', 'Capsicum', 'Chili', 'Potato', 'Tomato']
        Palm =  ['Dates', 'Coconut', 'Arecanut']
        Myrtle =  ['Jambun(Syzygium cumini)', 'Guava', 'Cloves']
        Ginger =  ['Cardamom', 'Ginger', 'Turmeric']
        Mallows =  ['Lady Finger', 'Cotton', 'Jute']
        Rutaceae =  ['Curry leaves', 'Lemon', 'Orange']
        Cashews =  ['Mango', 'Cashewnuts', 'Pistachio Nut']
        Rose =  ['Apricot', 'Apple', 'Almond Nut']
        Laurels =  ['Bay Leaf', 'Cinnamon']
        Onion =  ['Onion', 'Garlic']
        Mulberry =  ['Jackfruit', 'Figs']

        grasses1 = []
        cucurbits1 = []
        legumes1 = []
        umbellifers1 = []
        mustards1 = []
        nightshade1 = []
        palm1 = []
        myrtle1 = []
        ginger1 = []
        mallows1 = []
        rutaceae1 = []
        cashews1 = []
        rose1 = []
        laurels1 = []
        onions1 = []
        mulberry1 = []

        for i in crop_name:
            if i in Grasses:
                grasses1.append(i)
        for i in crop_name:
            if i in Cucurbits:
                cucurbits1.append(i)
        for i in crop_name:
            if i in Legumes:
                legumes1.append(i)
        for i in crop_name:
            if i in Umbellifers:
                umbellifers1.append(i)
        for i in crop_name:
            if i in Mustards:
                mustards1.append(i)
        for i in crop_name:
            if i in Nightshade:
                nightshade1.append(i)
        for i in crop_name:
            if i in Palm:
                palm1.append(i)
        for i in crop_name:
            if i in Myrtle:
                myrtle1.append(i)
        for i in crop_name:
            if i in Ginger:
                ginger1.append(i)
        for i in crop_name:
            if i in Mallows:
                mallows1.append(i)
        for i in crop_name:
            if i in Rutaceae:
                rutaceae1.append(i)
        for i in crop_name:
            if i in Cashews:
                cashews1.append(i)
        for i in crop_name:
            if i in Rose:
                rose1.append(i)
        for i in crop_name:
            if i in Laurels:
                laurels1.append(i)
        for i in crop_name:
            if i in Onion:
                onions1.append(i)
        for i in crop_name:
            if i in Mulberry:
                mulberry1.append(i)
        
        existing_list = dict_family[existing]
        if(existing_list == "Grasses"):
            grasses1 = []
        if(existing_list == "Cucurbits"):
            cucurbits1 = []
        if(existing_list == "Legumes"):
            legumes1 = []
        if(existing_list == "Umbellifers"):
            umbellifers1 = []
        if(existing_list == "Mustards"):
            mustards1 = []
        if(existing_list == "Nightshade"):
            nightshade1 = []
        if(existing_list == "Palm"):
            palm1 = []
        if(existing_list == "Myrtle"):
            myrtle1 = []
        if(existing_list == "Ginger"):
            ginger1 = []
        if(existing_list == "Mallows"):
            mallows1 = []
        if(existing_list == "Rutaceae"):
            rutaceae1 = []
        if(existing_list == "Cashews"):
            cashews1 = []
        if(existing_list == "Rose"):
            rose1 = []
        if(existing_list == "Laurels"):
            laurels1 = []
        if(existing_list == "Onion"):
            onions1 = []
        if(existing_list == "Mulberry"):
            mulberry1 = []


        return render_template('crop_prediction.html',existing=existing, grasses_c=grasses1, cucurbits_c=cucurbits1, legumes_c=legumes1, umbellifers_c=umbellifers1, mustards_c=mustards1, nightshade_c=nightshade1, palm_c=palm1, myrtle_c=myrtle1, ginger_c=ginger1, mallows_c=mallows1, rutaceae_c=rutaceae1, cashews_c=cashews1, rose_c=rose1, laurels_c=laurels1, onions_c=onions1, mulberry_c=mulberry1 ,display=True)
        # return render_template('crop_prediction.html',crops=crop_list ,crop_num = len(crop_list),display=True)

    return render_template('crop_prediction.html',error=0)



@app.route('/fertilizer_info',methods=['POST','GET'])
def fertilizer_info():
    data = pd.read_csv('final_fertilizer.csv')
    crops = data['Crop'].unique()

    if request.method == 'GET':
        crop_se = request.args.get('manager')
        query = data[data['Crop']==crop_se]
        query = query['query'].unique()
        queryArr = []
        if len(query):
            for query_name in query:
                queryObj = {}
                queryObj['name'] = query_name
                print(query_name)
                queryArr.append(queryObj)
            
            return jsonify({'data':render_template('fertilizer.html',crops=crops,crop_len=len(crops)),
                            'query':queryArr})
           
    
    if request.method == 'POST':
        crop_name = request.form['crop']
        query_type = request.form['query']
        query = data[data['Crop']==crop_name]
        answer = query[query['query']== query_type]
        answer = answer['KCCAns'].unique()
        protection = []
        for index in answer:
            protection.append(index)

        return render_template('fertilizer.html',protection=protection,protection_len=len(protection),display=True,crops=crops,crop_len=len(crops))


    return render_template('fertilizer.html',crops=crops,crop_len=len(crops),query_len=0)


@app.route('/shop',methods=['POST','GET'])
def shop():
    if request.method == 'POST':
        city = request.form['city']
        print(city)

        return render_template('fertilizer_shop.html',city=city,data=True)

    return render_template('fertilizer_shop.html')

@app.route('/feedback', methods=['POST','GET'])
def feedback():
    if request.method == 'POST':
        addcsv = []
        
        addcsv.append(request.form['cropname'])
        addcsv.append(request.form['cult'])
        addcsv.append(request.form['duration'])
        addcsv.append(request.form['location'])
        addcsv.append(request.form['seed'])
        addcsv.append(request.form['pesticides'])
        addcsv.append(request.form['quality'])
        addcsv.append(request.form['cost'])
        addcsv.append(request.form['yield'])
        addcsv.append(request.form['sold'])

        append_list_as_row('outcomes.csv', addcsv)

        return render_template('feedback_complete.html')

    return render_template('feedback.html')

@app.route('/support', methods=['GET'])
def support():
    return render_template('support.html')

@app.route('/register',methods=['POST','GET'])
def register():
    if request.method == 'POST':

        first_name = request.form['first_name']
        middle_name = request.form['middle_name']
        last_name = request.form['last_name']
        phone_number = request.form['phone']
        kisan_id = request.form['kisan_id']
        adhar_id = request.form['adhar_id']
        state = request.form['state']
        city = request.form['city']
        fullAddress = request.form['fullAddress']
        locality = request.form['locality']
        zipcode = request.form['zipcode']
        password = request.form['password']
        conform_password = request.form['conform_password']
        print(first_name,middle_name,last_name, phone_number,kisan_id,state,city,fullAddress,locality,password,conform_password)


        docs = db.collection(u'kisan_id').get()

       
            # print(u'{} => {}'.format(doc.id,data))

        if password == conform_password:
             for doc in docs:
                data = doc.to_dict()
                if data['id'] == kisan_id:
                    try:
                        email_id = 'kisan'+kisan_id+'@gmail.com'
                        user = auth.create_user(email = email_id,password= password)
                        print('Sucessfully created new user: {0}'.format(user.uid))
                    except :
                        return render_template('register.html',alert=2,first_name=first_name)
                    

                    

                    if user.uid:
                        doc_ref = db.collection(u'users').document(u''+user.uid)
                        doc_ref.set({
                            u'first_name': first_name,
                            u'middle_name': middle_name,
                            u'last_name' :last_name,
                            u'phone_number': phone_number,
                            u'kisan_id': kisan_id,
                            u'adhar_id': adhar_id,
                            u'state': state,
                            u'city': city,
                            u'fullAddress': fullAddress,
                            u'locality': locality,
                            u'zipcode': zipcode
                            })

                        
                        return render_template('register.html',alert=1,first_name=first_name)

    return render_template('register.html')


@app.route('/login',methods=['POST','GET'])
def login():

    if request.method == 'POST':
        login_kisan = Login()
        data,email = login_kisan.kisan_login()
        print(data)
        print(type(data))
        
        if data == 'successful':
            user = auth.get_user_by_email(email)
            print('Successfully fetched user data: {0}'.format(user.uid))

            doc_ref = db.collection(u'users').document(u''+user.uid)
            
            docs = doc_ref.get().to_dict()
            print(docs)
            print(user.uid)
            return render_template('kisan_profile.html',data=docs,display=False,user_id=user.uid)
        else:
            flash(f'Login Failed Please check Your Kisan ID Number and Password','danger')
            return redirect('/login')



    return render_template('login.html')


@app.route('/add_data/<id>',methods=['POST','GET'])
def add_data(id):
    print(id)
    if request.method == 'POST':
        crop_1 = request.form['crop_1']
        crop_2 = request.form['crop_2']
        crop_3 = request.form['crop_3']
        crop_4 = request.form['crop_4']
        print(crop_1,crop_2,crop_3)

        db_ref =  db.collection(u'users').document(u''+id)
        print(db_ref.id)
        
        db_ref.update({
			u'crop_1':crop_1,
			u'crop_2':crop_2,
            u'crop_3':crop_3,
            u'crop_4':crop_4
			})
        docs = db_ref.get().to_dict()
        # flash(f'New Data Added!','success')
        # return redirect('/login')
        user_id = db_ref.id
        print(user_id)
        print("comes here")
        flash(f'Data Updated!','success')
        return render_template('kisan_profile.html',data=docs,dispaly=True,user_id=user_id)

    return render_template('add_data.html',user_id=id)


@app.route('/issue/<user_id>', methods=['GET','POST'])
def issue(user_id):
	if request.method == 'POST':
			fullName = request.form['fullName']
			issue = request.form['issue']

			doc_ref = db.collection(u'issue').document(u''+user_id).collection(u'user_issue').document()
			doc_ref.set({
				u'fullName': fullName,
				u'issue': issue,
				u'seen' : 0
			})

			return render_template('issue.html', user_id = user_id,data=True)
	return render_template('issue.html',user_id = user_id,data=False)



@app.route('/check_issue',methods=['POST','GET'])
def check_issue():
	
	docs = db.collection(u'issue').get()
	# print(docs.to_dict())
	lt = []
	ids = []
	for doc in docs:
		dt = {}
		print(u'{} => {}'.format(doc.id, doc.to_dict()))
		dt['id'] = doc.id
		dt['data'] = doc.to_dict()
		lt.append(dt)
		ids.append(doc.id)
		# print(doc.to_dict())
	# print(len(lt))
	print(lt)
	print(ids)

	data = []
	for i in range(len(ids)):
		id = ids[i]
		docs = db.collection(u'issue').document(u''+id).collection(u'user_issue').get()
		print('comes here')
		for doc in docs:
			dt = {}
			print(u'{} => {}'.format(doc.id, doc.to_dict()))
			dt['user_id'] = id
			dt['id'] = doc.id
			dt['data'] = doc.to_dict()
			data.append(dt)
	
	print(data)

	# print(lt[0]['data']['seen'])
	# print(lt[1]['id'])
	
	# print(doc)
	return render_template('check_issue.html',data=data,data_len=len(data))
	# return 'daata'


	
@app.route('/submit_issue/<user_id>/<data_id>',methods=['POST','GET'])
def submit_issue(user_id,data_id):
    
    docs = db.collection(u'issue').document(u''+user_id).collection(u'user_issue').document(u''+data_id).get().to_dict()
    print(docs)
    print(user_id,data_id)
    if request.method == 'POST':
        db_ref =  db.collection(u'issue').document(u''+user_id).collection(u'user_issue').document(u''+data_id)
        answer = request.form['answer']
        print(answer)
        db_ref.update({u'answer': answer,u'seen':1})

        account_sid = 'AC7b12fbd4c6a2cce3b4fa7d049dc074a7'
        auth_token = '5eb6f4ccbb2dcc0e0f71a6ae165a8cd4'
        
        client = Client(account_sid, auth_token)
        message = client.messages \
        .create(body="Your issue answer send to your profile check your profile for futher update",
            from_='+15104038027',
            to='+919663077540')
        
        print(message.sid)
        
        flash(f'Answer Submited!','success')
        return redirect(url_for('check_issue'))
        
    print(user_id,data_id)
    return render_template('submit_issue.html',data=docs,user_id=user_id,data_id=data_id)
	# return 'done'


@app.route('/issue_update/<user_id>',methods=['POST','GET'])
def issue_update(user_id):
    # docs = db.collection(u'issue').document(u''+id)
    docs =  db.collection(u'issue').document(u''+user_id).collection(u'user_issue').get()
    print(user_id)
    print(docs)
    lt = []
    for doc in docs:
        dt = {}
        print(u'{} => {}'.format(doc.id, doc.to_dict()))
        dt['id'] = doc.id
        dt['data'] = doc.to_dict()
        lt.append(dt)

    print(lt)



    return render_template('issue_update.html',data=lt,data_len = len(lt))
    # return 'data'

@app.route('/admin_login',methods=['POST','GET'])
def admin_login():
    if request.method == 'POST':
        login_kisan = Login_Admin()
        data,email = login_kisan.admin_login()
        print(data)
        print(type(data))
        
        if data == 'successful':
            user = auth.get_user_by_email(email)
            print('Successfully fetched user data: {0}'.format(user.uid))

            doc_ref = db.collection(u'users').document(u''+user.uid)
            
            docs = doc_ref.get().to_dict()
            print(docs)
            print(user.uid)
            return render_template('admin.html')
        else:
            flash(f'Login Failed Please check Your Kisan ID Number and Password','danger')
            return redirect('/admin_login')



    return render_template('admin_login.html')

@app.route('/kisan_center',methods=['POST','GET'])
def kisan_center():
    if request.method == 'POST':
        login_kisan = Login_Kisan()
        data,email = login_kisan.kisan_center_login()
        print(data)
        print(type(data))
        
        if data == 'successful':
            user = auth.get_user_by_email(email)
            print('Successfully fetched user data: {0}'.format(user.uid))

            doc_ref = db.collection(u'users').document(u''+user.uid)
            
            docs = doc_ref.get().to_dict()
            print(docs)
            print(user.uid)
            return render_template('kisan_center.html',data=False,error=False)
        else:
            flash(f'Login Failed Please check Your Kisan ID Number and Password','danger')
            return redirect('/kisan_center')

    return render_template('kisan_login.html')


@app.route('/add_kisan_id',methods=['POST','GET'])
def add_kisan_id():
    if request.method == 'POST':
        kisan_id = request.form['kisan_id']
        # print(kisan_id)
        if len(kisan_id) != 13:
            print(kisan_id)
            return render_template('kisan_center.html',data=False,error=True)

        doc_ref = db.collection(u'kisan_id').document()
        doc_ref.set({
            u'id': kisan_id
            })

        return render_template('kisan_center.html',data=True,error=False)



if __name__ == "__main__":
    
    app.run(debug=True)
    
# tensorflow = 1.15.0
# keras = 2.0.8