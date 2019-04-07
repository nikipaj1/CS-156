import numpy as np
import random

class DataManipulation:
    def __init__(self, data_set, point1 = None, point2 = None):
        self.data_set = data_set
        if point1 is None:
            self.point1 = [random.uniform(-1,1), random.uniform(-1,1)]
        else: 
            self.point1 = point1
        if point2 is None:
            self.point2 = [random.uniform(-1,1), random.uniform(-1,1)]
        else: 
            self.point2 = point2
    
    def train_test_split(self, train = 0.5):
        self.X_train = random.sample(self.data_set, int(train*(len(self.data_set))))
        self.X_test = [x for x in self.data_set if x not in self.X_train]
    
    def target(self, point):
        x1,y1 = self.point1
        x2,y2 = self.point2
        
        slope = (y2-y1)/(x2-x1)
        x,y = point[1], point[2]
        
        if y > slope * (x-x1) + y1: return 1
        else: return -1 
        
    def hypothesis(self,point):
        return np.sign(np.dot(point, self.w))
    
    def test(self, E_in = True, noise = None):
        error_count = 0
               
        if E_in is False:
            target_test = [self.target(i) for i in self.X_test]
            hypothesis = np.array([[self.hypothesis(i)] for i in self.X_test])
        else:
            target_test = [self.target(i) for i in self.X_train]
            hypothesis = np.array([[self.hypothesis(i)] for i in self.X_train])
        
        if noise is not None:
            hypothesis = self.simulate_noise(hypothesis, noise)
                     
        for j in range(len(target_test)):
            if target_test[j] != hypothesis[j,0]:
                error_count += 1
                
        return error_count
    
    def simulate_noise(self, data, fraction):
        switch_number = data.size * fraction
        switched = set()
        
        while len(switched) < switch_number:
            index = random.randint(0, data.size - 1)
            
            data[index,0] *= -1
            
            switched.add(index)
            
        return data
    
class PLA(DataManipulation):
    def __init__(self,data_set, w = None, point1 = None, point2 = None):
        if w is None: 
            self.w = np.zeros((3,1))
        else:
            self.w = w
        super().__init__(data_set, point1, point2)
                         
    def train(self):
        repeats = 0
        misclassified = []
        target = [self.target(i) for i in self.X_train]
        while True:
            repeats += 1
            for i in range(len(self.X_train)):
                hypothesis = self.hypothesis(self.X_train[i])
                if target[i] != hypothesis:
                    misclassified.append([self.X_train[i], target[i]])         
            if not misclassified:
                break            
            random_miss = random.choice(misclassified)
            self.w += random_miss[1] * np.array([random_miss[0]]).T           
            misclassified = []
        return repeats
        
class LinearRegression(DataManipulation):
    def __init__(self, data_set, w = None):
        if w is None: 
            self.w = np.zeros((3,1))
        else:
            self.w = w
        super().__init__(data_set)
        
    def train(self, noise = None, non_linear = False):
        X_matrix = np.array(self.X_train)
        
        if not non_linear:
            y_vector = np.array([[self.target(x)] for x in self.X_train])
        else:
            y_vector = np.array([[self.target_nonlinear(x)] for x in self.X_train])
               
        if noise is not None:
            y_vector = self.simulate_noise(y_vector, 0.1)
        
        self.w = np.dot(np.linalg.pinv(X_matrix), y_vector)
        
    def target_nonlinear(self, point):
        return np.sign(point[1]**2 + point[2]**2 - 0.6)
    
    def test_nonlinear(self, E_in = True, noise = None):
        error_count = 0
               
        if E_in is False:
            target_test = [self.target_nonlinear(i) for i in self.X_test]
            hypothesis = np.array([[self.hypothesis(i)] for i in self.X_test])
        else:
            target_test = [self.target_nonlinear(i) for i in self.X_train]
            hypothesis = np.array([[self.hypothesis(i)] for i in self.X_train])
        
        if noise is not None:
            hypothesis = self.simulate_noise(hypothesis, noise)
                     
        for j in range(len(target_test)):
            if target_test[j] != hypothesis[j,0]:
                error_count += 1
                
        return error_count   

    def train_real_data(self, k = None):
        X_matrix = np.array(self.X_train)
        y_vector = np.array([self.y_train]).T
        
        self.w = np.dot(np.linalg.pinv(X_matrix), y_vector)
        
        # regularization 
        if k is not None:
            lambd = 10**k
                        
            I_matrix = np.identity(len(self.X_train[0]))
            inverse = np.linalg.inv(np.dot(np.array(self.X_train).T, (np.array(self.X_train))) + lambd * I_matrix)
            pseudo_inv = np.dot(inverse, np.array(self.X_train).T)
            
            self.w = np.dot(pseudo_inv, np.array([self.y_train]).T)
                       
    def test_real_data(self, E_in = False):
        
        error_count = 0
        
        if E_in:
            target_test = self.y_train
            hypothesis = np.array([[self.hypothesis(point)] for point in self.X_train])
        else:
            target_test = self.y_test
            hypothesis = np.array([[self.hypothesis(point)] for point in self.X_test])
        
        for j in range(len(target_test)):
            if target_test[j] != hypothesis[j]:
                error_count += 1
        return error_count
        
        
            
    
    