from sklearn.preprocessing import LabelEncoder
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import pandas as pd
import joblib
import json
from flask import Flask
from flask_restx import Api

app = Flask(__name__)
api = Api(app)


class MLModels:
    def __init__(self):
        self.model_info = {}
        self.models_type = [{'name': 'RandomForestClassifier', 'task': 'classification'},
                            {'name': 'lightgbm', 'task': 'classification'},
                            {'name': 'DecisionTreeRegressor', 'task': 'regression'}]

    def preprocessing(self, data):
        """
        Функция для предобработки входных данных (удаление null значений и преобразование
        категориальных значений в числовые)
        :param data: данные, которые необходимо предобработать
        :return: предобработанные данные
        """
        data = data.dropna()
        cat_features = data.select_dtypes(include=['object']).columns.tolist()
        if cat_features is not None:
            for col in cat_features:
                data.loc[:, col] = LabelEncoder().fit_transform(data.loc[:, col])

        return data

    def model_train(self, input_data_train):
        """
        В этой функции проходит обучение модели, сохранение обученных моделей в ./model и
        формирование отчета по обученным моделям (models_info.txt).
        Структура отчета:
        (
        dataset: на каком датасете обучалась модель
        features: фичи на которых обучаласть модель
        target: целевая переменная
        metric: метрика качества
        model_name: тип модели (RandomForestClassifier, lightgbm,...)
        name: имя модели как ее назвал пользователь
        exist_models: список всех обученных моделей
        exist_models_type: доступные классы моделей
        models_description: характеристики всех обученных моделей
        )
        :param input_data_train: Данные для обучения (передаются пользователем по API)
        :return: Возвращает эксземпляр класса
        """
        features = input_data_train['features']
        target = input_data_train['target']
        data = pd.read_json(input_data_train['data'])
        params = input_data_train['params']
        model_name = input_data_train['model_name']
        dataset = input_data_train['dataset']
        name = input_data_train['name']

        # Вибираем тип модели, который планируется обучить
        if model_name == 'RandomForestClassifier':
            self.model = RandomForestClassifier(**params)
            metric = accuracy_score
        elif model_name == 'lightgbm':
            self.model = lgb.LGBMClassifier(**params)
            metric = accuracy_score
        elif model_name == 'DecisionTreeRegressor':
            self.model = DecisionTreeRegressor(**params)
            metric = mean_squared_error

        # Если модель есть в списке обученных моделей, то обучим ее еще раз
        if f'{name}.pkl' in os.listdir('model'):
            self.model = joblib.load(f'model/{name}.pkl')
        # Проводим предобработку обучающего датасета
        train_df = self.preprocessing(data)
        # Разбиваем на train/test
        X_train, X_test, y_train, y_test = train_test_split(train_df[features], train_df[target],
                                                            test_size=0.2, random_state=42)
        # Обучаем модель
        self.model.fit(X_train, y_train.values.ravel())
        # Получаем качество модели в зависимости от класса модели
        _metric = metric(y_test, self.model.predict(X_test))
        # Собираем характеристики модели в словарь
        self.model_info['name'] = f'{name}.pkl'
        self.model_info['model_name'] = model_name
        self.model_info['metric'] = _metric
        self.model_info['dataset'] = dataset
        self.model_info['features'] = features
        self.model_info['target'] = target

        # Сохраняем обученную модель в ./model
        joblib.dump(self.model, f'model/{name}.pkl')

        # Записываем характеристики модели в ./models_info.txt
        with open('models_info.txt', 'a') as f:
            f.write(json.dumps(self.model_info))
            f.write('\n')

        # Если обученная ранее модель была удалена, то удаляем ее из отчета по доступным моделям
        if os.path.exists('models_info.txt'):
            self.update_txt_file()

        return self


    def update_txt_file(self):
        """
        Функция для обновления ./models_info.txt (это необходимо, так как некоторые модели
        могут быть удалены и хранить по ним информацию нет смысла)
        :return: Сообщение о том, что файл обновлен
        """
        path = r'models_info.txt'
        with open(path) as file:
            lines = file.readlines()
            lines = [json.loads(line) for line in lines]
        os.remove(path)
        for line in lines:
            with open('models_info.txt', 'a') as f:
                if line['name'] in os.listdir('model'):
                    f.write(json.dumps(line))
                    f.write('\n')

        return 'file was update'

    def get_models_info(self):
        """
        Функция возвращает json с отчетом, который формировали в model_train.
        То что возвращает эта функция и будет то, что получит пользаватель, когда
        сделает requests.get('http://127.0.0.1:5000/api/ml_models/train')
        :return: json с отчетом для пользователя
        """
        path = r'models_info.txt'
        with open(path) as file:
            lines = file.readlines()
            lines = [json.loads(line) for line in lines]

        exist_model_list = os.listdir('model')

        models_info = [{'model_info(train)': self.model_info},
                       {'exist_models': exist_model_list},
                       {'exist_models_type': self.models_type},
                       {'models_description': lines}]

        return models_info

    def available_models_info(self):
        path = r'models_info.txt'
        with open(path) as file:
            lines = file.readlines()
            lines = [json.loads(line) for line in lines]
        models_info = {'models_description': lines}

        return models_info

    def available_сlass_model(self):
        return {'class of models': self.models_type}

    def predict_model(self, input_data_predict, name):
        """
        Функция возвращает предсказание модели по данным, которые отправил пользователь.
        Будет использоваться в PUT запросе для получения предсказаний
        :param input_data_predict: данные, который отправил пользователь по API
        :param name: название модели, которая будет использоваться для предсказаний. Все доступные
        имена лежат в ./models
        :return: json с предсказаниями
        """
        try:
            data = pd.read_json(input_data_predict['data'])
            # Предобрабатываем входные данные
            test_df = self.preprocessing(data)
            # Импорт модели, которую будем использовать
            model = joblib.load(f'model/{name}.pkl')
            # Делаем предсказание
            self.pred = model.predict(test_df)
            self.res = {'predict': self.pred.tolist()}
            return {'predict': self.pred}
        except:
            self.pred = 'this model doesnt exist or invalid input data'
            self.res = {'predict': self.pred}
            return {'predict': self.pred}

    def get_model_predict(self):
        """
        Функция будет использаваться в GET запросе от пользователя для получения предсказаний
        :return: Возвращает предсказание модели из predict_model
        """
        return self.res

    def delete_model(self, name):
        """
        Функция используется для удалния модели из доступных моделей, которые лежат в ./model
        :param name: Имя модели, которую требуется удалить
        :return: message
        """
        try:
            os.remove(f'model/{name}.pkl')
            return 'successfully deleted'
        except:
            return 'file doesnt exist'



model = MLModels()
