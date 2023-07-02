import streamlit as st
from PIL import Image
import pandas as pd
from pickle import load
import numpy as np

first_img = Image.open('best-drag-races-montage-1-lead.jpg')

st.set_page_config(
    layout='wide',
    page_title='',
    page_icon=first_img)

st.title('Автоценность')
st.image(first_img,width=800)

st.sidebar.header('Параметры автомобиля:')
df = pd.read_csv('clean_data.csv')

car_brand = st.sidebar.selectbox('Марка',(sorted(df['brand'].unique())))
car_model = st.sidebar.selectbox('Модель',(sorted(df[df['brand']==car_brand]['model'].unique())))

car_fuel = st.sidebar.selectbox('Вид топлива',(df[(df['brand']==car_brand)&
                                                  (df['model']==car_model)]['fuel'].unique()))

car_year = st.sidebar.slider('Год выпуска', min_value=int(df['year'].min().astype(int)),
                             max_value=int(df['year'].max()),
                             value=int(df['year'].min().astype(int)), step=1)

car_mileage = st.sidebar.slider('Пробег в тыс.км.', min_value=0,
                             max_value=int(df['km_driven'].max()/1000),
                             value=0, step=1)

car_transmission = st.sidebar.selectbox('Трансмиссия (0 = Manual, 1 = Auto)',(df[(df['brand']==car_brand)&
                                                          (df['model']==car_model)&
                                                          (df['fuel']==car_fuel)]['transmission'].mode()))

car_owner = st.sidebar.slider('Количество владельцев', min_value=0,
                             max_value=int(df['owner'].max()),
                             value=0, step=1)

car_seller = st.sidebar.selectbox('У кого хотите преобрести автомобиль?',
                                  sorted(df['seller_type'].unique()))

guest_data = {
'brand' : car_brand,
'model' : car_model,
'year' : car_year,
'km_driven' : car_mileage,
'fuel' : car_fuel,
'seller_type' : car_seller,
'transmission' : car_transmission,
'owner' : car_owner,
'torque' : df[(df['brand']==car_brand)&(df['model']==car_model)&(df['fuel']==car_fuel)&(df['transmission']==car_transmission)]['torque'].mean(),
'seats' : df[(df['brand']==car_brand)&(df['model']==car_model)&(df['fuel']==car_fuel)&(df['transmission']==car_transmission)]['seats'].mean(),
'consumption' : df[(df['brand']==car_brand)&(df['model']==car_model)&(df['fuel']==car_fuel)&(df['transmission']==car_transmission)]['consumption'].mean(),
'volume' : df[(df['brand']==car_brand)&(df['model']==car_model)&(df['fuel']==car_fuel)&(df['transmission']==car_transmission)]['volume'].mean(),
'power' : df[(df['brand']==car_brand)&(df['model']==car_model)&(df['fuel']==car_fuel)&(df['transmission']==car_transmission)]['power'].mean()
}
res_df = pd.DataFrame(guest_data,index=[0])




st.write('''
## Задумали купить или продать свой автомобиль?
Если вы хотите купить или продать свой автомобиль, то вопрос стоимости будет едва-ли не самым важным.\n
Рынок заполнен огромным количеством предложений, просмотреть все их и понять какая стоимость реальная очень сложно.\n
Даже если вы не разбираетесь в автомобилях, наша команда сможет вам помочь!\n
Автоценность постоянно мониторит авторынок и знает стоимость и параметры любого автомобиля.\n
На основе этих знаний искусстенный интелект определяет реальную стоимость.\n
В расчет берутся все параметры: от марки авто и его пробега, до положения нашей планеты в космическом пространстве.\n
Вы можете выбрать автомобиль и его параметры в меню выбора параметров, оно находится слева,\n
и мы скажем вам со 100% точностью его стоимость!''')

st.write(res_df)

with open('model.pickle','rb') as file:
    model = load(file)
with open('encoder.pickle','rb') as file:
    encoder = pd.read_pickle(file)
with open('scal.pickle','rb') as file:
    scal = load(file)

encoded_res = encoder.transform(res_df)

scaled_res = scal.transform(encoded_res)

overall_pred = np.exp(model.predict(scaled_res))
st.write(f'Стоимость автомобиля вашей мечты: {round(overall_pred[0])}')

