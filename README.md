# ml_car_price_prediction_service

Создание сервиса по предсказанию стоимости автомобиля

В этом проекте мы предсказываем стоимость автомобиля с помощью машинного обучения и деплоим его через сервис Streamlit

Ссылка на сайт с готовым [сервисом](https://ml-car-price-prediction.streamlit.app/)

## Файлы в репозитории:

- best-drag-races-montage-1-lead.jpg : Просто картинка с интернета
- car_price_prediction.ipynb : файл с EDA и обученнием модели
- car_price_prediction_service.py : код для запуска сайта на стримлите
- cars.csv : файл с грязными файлами
- clean_data.csv : файл с предобработанными файлами
- encoder.pickle, model.pickle, scal.pickle : обученные энкодер,скалер и модель
- requirements.txt : нужные версии библиотек 
# Исследовтельский анализ данных

- Убрали явные и неявные дубликаты
- Из строковых колонок достали нужные данные
- Привели к нежным форматам данных
- Все что можно категориальные порядковые значения замапали

# Ваш запуск сайта

Если хотите запустить сервис самостоятельно, то нужно создать репозиторий со всеми файлами и запустить через стримлит.

Удачи и спасибо за внимание!
