import math
import random
import numpy as np
import matplotlib.pyplot as plt
random.seed(0)


def rand(a, b):
    return (b - a) * random.random() + a


def make_matrix(m, n, fill=0.0):
    mat = []
    for i in range(m):
        mat.append([fill] * n)
    return mat


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


class BPNeuralNetwork:
    def __init__(self):
        self.input_n = 0
        self.hidden_n = 0
        self.output_n = 0
        self.input_cells = []
        self.hidden_cells = []
        self.output_cells = []
        self.input_weights = []
        self.output_weights = []
        self.input_correction = []
        self.output_correction = []
        self.error_list=[]

    def setup(self, ni, nh, no):
        self.input_n = ni+1
        self.hidden_n = nh
        self.output_n = no
        # init cells
        self.input_cells = [1.0] * self.input_n
        self.hidden_cells = [1.0] * self.hidden_n
        self.output_cells = [1.0] * self.output_n
        # init weights
        self.input_weights = make_matrix(self.input_n, self.hidden_n)
        self.output_weights = make_matrix(self.hidden_n, self.output_n)
        # random activate
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                self.input_weights[i][h] = rand(-0.2, 0.2)
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                self.output_weights[h][o] = rand(-2.0, 2.0)
        # init correction matrix
        self.input_correction = make_matrix(self.input_n, self.hidden_n)
        self.output_correction = make_matrix(self.hidden_n, self.output_n)

    def predict(self, inputs):
        # activate input layer
        for i in range(self.input_n - 1):
            inputc=inputs[i]/255.0
            self.input_cells[i] = inputc
            
        # activate hidden layer
        for j in range(self.hidden_n):
            total = 0.0
            for i in range(self.input_n):
                total += self.input_cells[i] * self.input_weights[i][j]
            self.hidden_cells[j] = sigmoid(total)
        # activate output layer
        for k in range(self.output_n):
            total = 0.0
            for j in range(self.hidden_n):
                total += self.hidden_cells[j] * self.output_weights[j][k]
            self.output_cells[k] = sigmoid(total)
        return self.output_cells[:]

    def back_propagate(self, case, label, learn, correct):
        # feed forward
        self.predict(case)
        # get output layer error
        output_deltas = [0.0] * self.output_n
        outputc=np.argmax(self.output_cells)
        labeltmp=np.zeros(10)
        labeltmp[outputc]=1
        for o in range(self.output_n):
            error = labeltmp[o] - self.output_cells[o]
            output_deltas[o] = sigmoid_derivative(self.output_cells[o]) * error
        # get hidden layer error
        hidden_deltas = [0.0] * self.hidden_n
        for h in range(self.hidden_n):
            error = 0.0
            for o in range(self.output_n):
                error += output_deltas[o] * self.output_weights[h][o]
            hidden_deltas[h] = sigmoid_derivative(self.hidden_cells[h]) * error
        # update output weights
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                change = output_deltas[o] * self.hidden_cells[h]
                self.output_weights[h][o] += learn * change + correct * self.output_correction[h][o]
                self.output_correction[h][o] = change
        # update input weights
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                change = hidden_deltas[h] * self.input_cells[i]
                self.input_weights[i][h] += learn * change + correct * self.input_correction[i][h]
                self.input_correction[i][h] = change
        # get global error
        error = 0.0
        for o in range(len(labeltmp)):
            error += 0.5 * (labeltmp[o] - self.output_cells[o]) ** 2
        return error

    def train(self, cases, labels, limit=100, learn=0.05, correct=0.1,batchsize=100):
        datas =np.insert(cases, 2025, values=labels, axis=1)
        for i in range(limit):
            error = 0.0
            np.random.shuffle( datas )
            data_batch=datas[:batchsize,:-1]
            case_batch = datas[:batchsize, -1:]

            print ("Train Time: ",i)           
            for j in range(batchsize):
                case = data_batch[j]
                label = case_batch[j]
                error += self.back_propagate(case, label, learn, correct)
                
            
            '''
        for j in range(limit):
            error = 0.0
            for i in range(batchsize):
                if i%1000==0:
                    print (j,"Train Time: ",i/1000)
                    label = labels[i]
                    case = cases[i]
                    error += self.back_propagate(case, label, learn, correct)
            self.errorlist.append(error)
            '''
    def test(self,train_data,train_label,test_data,test_label):

        self.setup(2025, 20, 10)
        self.train(train_data, train_label, 10000, 0.1, 0.05,100)
        accuracy=[0.0]*10000
        i=0
        prediction=[0.0]*10000
        for case in test_data:
            prediction[i]=np.argmax(self.predict(case))
            if prediction[i]==test_label[i]:
                accuracy[i]=1
            else:
                accuracy[i]=0
            i+=1
            print (i)
        print("Accuracy:",1-np.mean(accuracy))
        return prediction


if __name__ == '__main__':
    data = np.fromfile("train\mnist_train_data",dtype=np.uint8)
    data=data.reshape(60000,45,45)
    train_data=data.reshape(60000,2025)
    train_label = np.fromfile("train\mnist_train_label",dtype=np.uint8)
    data = np.fromfile("test\mnist_test_data",dtype=np.uint8)
    data=data.reshape(10000,45,45)
    test_data =data.reshape(10000,2025)
    test_label = np.fromfile("test\mnist_test_label",dtype=np.uint8)
    print test_data.shape
    nn = BPNeuralNetwork()
    pre=nn.test(train_data,train_label,test_data,test_label)
    plt.plot(nn.error_list)