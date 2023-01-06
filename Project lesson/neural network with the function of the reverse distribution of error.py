import numpy as np
import sys

class PartyNN(object):
    def __init__(self, learning_rate = 0.1):
        # веса от нулевого слоя к первому
        self.weights_0_1 = np.random.normal(0.0, 2**-0.5, (2, 3))
        # веса от первого слоя к последнему (случайно заданные как и для первого)
        self.weights_1_2 = np.random.normal(0.0, 1, (1, 2))
        # к каждому эллементу применяет сигмоидную функцию
        self.sigmoid_mapper = np.vectorize(self.sigmoid)
        self.learning_rate = np.array(learning_rate)

    # функция активации
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # метод прямого предсказания
    def predict(self, inputs):
        # начальные данные умножаем на импуты
        inputs_1 = np.dot(self.weights_0_1, inputs)
        # прогоняем через сигмоидную функцию
        outputs_1 = self.sigmoid_mapper(inputs_1)

        # умножаем вызодные значения из скрытого слоя на веса
        inputs_2 = np.dot(self.weights_1_2, outputs_1)
        # прогоняем через сигмоидную функцию
        outputs_2 = self.sigmoid_mapper(inputs_2)
        return outputs_2

    # метод тренироки
    def train(self, inputs, expected_predict):
        inputs_1 = np.dot(self.weights_0_1, inputs)
        outputs_1 = self.sigmoid_mapper(inputs_1)

        inputs_2 = np.dot(self.weights_1_2, outputs_1)
        outputs_2 = self.sigmoid_mapper(inputs_2)
        # сохраняются значения выходные предыдущего обхода
        actual_predict = outputs_2[0]
        # все что выше повторяется как в def predict

        # ошибка на выходном слое (данные - ожидаемые значения)
        error_layer_2 = np.array([actual_predict - expected_predict])
        # высчитывается градиент на выходном слое (дифференциал по dx от функции активации)
        gradient_layer_2 = actual_predict * (1 - actual_predict)
        # высчитывается дельта параметр весов на который буду прибавлятся значения весов НЕПОНЯТНО
        weights_delta_layer_2 = error_layer_2 * gradient_layer_2
        # изменения для весов от скрытого слоя к последним НЕПОНЯТНО
        self.weights_1_2 -= (np.dot(weights_delta_layer_2, outputs_1.reshape(1, len(outputs_1)))) * self.learning_rate

        # методом обратного расспространения делается тоже самое, но для скрытого уровня
        error_layer_1 = weights_delta_layer_2 * self.weights_1_2
        gradient_layer_1 = outputs_1 * (1 - outputs_1)
        weights_delta_layer_1 = error_layer_1 * gradient_layer_1
        self.weights_0_1 -= np.dot(inputs.reshape(len(inputs), 1), weights_delta_layer_1).T * self.learning_rate

# метод оценки качества по среднему квадратическому отклонению (корню дисперсии, или матожидание)
def MSE(y, Y):
    return np.mean((y-Y)**2)

# тренировочные данные, где первые данные массив из входных еллементов на сеть, 
# второе - что должна предсказать
train = [
    ([0, 0, 0], 0),
    ([0, 0, 1], 1),
    ([0, 1, 0], 0),
    ([0, 1, 1], 0),
    ([1, 0, 0], 1),
    ([1, 0, 1], 1),
    ([1, 1, 0], 0),
    ([1, 1, 1], 1)
]

# # здесь происходит обучение

# сколько раз прогоняется через все данные тренировочные
epochs = 4000
# то насколько быстро сеть должна сдвигаться в ту сторону обучения которую выбрала
# пирмерно должен быть между 0 и 0.2
learning_rate = 0.05

# создаем сеть
network = PartyNN(learning_rate=learning_rate)

# в каждой эпохе проходим нейронкой все значения train 
for e in range(epochs):
    inputs_ = []
    correct_predictions = []
    for input_stat, correct_predict in train:
        # передаем входные данные и ожидаемые
        network.train(np.array(input_stat), correct_predict)
        # сохраняем значения
        inputs_.append(np.array(input_stat))
        correct_predictions.append(np.array(correct_predict))

    # оцениваем качество
    train_loss = MSE(network.predict(np.array(inputs_).T), np.array(correct_predictions))
    sys.stdout.write("\rProgress: {}, Training loss: {}".format(str(100 * e/float(epochs))[:4], str(train_loss)[:5]))

# после того как натренировали проверяем работу
for input_stat, correct_predict in train:
    print("For input: {} the predictions is: {}, expected: {}".format(
        str(input_stat),
        str(network.predict(np.array(input_stat)) > 0.5),
        str(correct_predict == 1)
    ))

for input_stat, correct_predict in train:
    print("For input: {} the predictions is: {}, expected: {}".format(
        str(input_stat),
        str(network.predict(np.array(input_stat))),
        str(correct_predict == 1)
    ))

# значения весов нейронов (какую либо информацию отсюда не получить)
# print(network.weights_0_1)
# print(network.weights_1_2)