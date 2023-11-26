import numpy as np

class SimplifiedBaggingRegressor:
    def __init__(self, num_bags, oob=False):
        self.num_bags = num_bags
        self.oob = oob
        
    def _generate_splits(self, data: np.ndarray):
        '''
        Generate indices for every bag and store in self.indices_list list
        '''
        self.indices_list = []
        data_length = len(data)
        for bag in range(self.num_bags):
            indices = np.random.randint(0, data_length, data_length).tolist()
            self.indices_list += [indices]
        
    def fit(self, model_constructor, data, target):
        '''
        Fit model on every bag.
        Model constructor with no parameters (and with no ()) is passed to this function.
        
        example:
        
        bagging_regressor = SimplifiedBaggingRegressor(num_bags=10, oob=True)
        bagging_regressor.fit(LinearRegression, X, y)
        '''
        self.data = None
        self.target = None
        self._generate_splits(data)
        assert len(set(list(map(len, self.indices_list)))) == 1, 'All bags should be of the same length!'
        assert list(map(len, self.indices_list))[0] == len(data), 'All bags should contain `len(data)` number of elements!'
        self.models_list = []
        num = 0
        for bag in range(self.num_bags):
            model = model_constructor()
            data_bag, target_bag = data[self.indices_list[num]], target[self.indices_list[num]]
            self.models_list.append(model.fit(data_bag, target_bag)) # store fitted models here
            num += 1
        if self.oob:
            self.data = data
            self.target = target
        
    def predict(self, data):
        '''
        Get average prediction for every object from passed dataset
        '''
        predicts = []
        for model in self.models_list:
            predicts += [model.predict(data)]
            
        avg_preds = np.sum(predicts, axis=0) / len(predicts)
            
        return avg_preds
     
    def _get_oob_predictions_from_every_model(self):
        '''
        Generates list of lists, where list i contains predictions for self.data[i] object
        from all models, which have not seen this object during training phase
        '''
        list_of_predictions_lists = [[] for _ in range(len(self.data))]
        for i in range(len(self.data)):
            preds = []
            for j in range(len(self.indices_list)):
                if i not in self.indices_list[j]:
                    preds += [self.models_list[j].predict([self.data[i]])]
            list_of_predictions_lists[i] = preds
                    
        self.list_of_predictions_lists = np.array(list_of_predictions_lists, dtype=object)
    
    def _get_averaged_oob_predictions(self):
        '''
        Compute average prediction for every object from training set.
        If object has been used in all bags on training phase, return None instead of prediction
        '''
        self._get_oob_predictions_from_every_model()
        avg_preds = []
        for i in range(len(self.list_of_predictions_lists)):
            if(len(self.list_of_predictions_lists[i]) == 0 or len(self.list_of_predictions_lists[i]) == len(self.models_list)):
                avg_preds += [None]
            else:
                avg_preds += [np.sum(self.list_of_predictions_lists[i]) / len(self.list_of_predictions_lists[i])]
        self.oob_predictions = avg_preds
        
        
    def OOB_score(self):
        '''
        Compute mean square error for all objects, which have at least one prediction
        '''
        self._get_averaged_oob_predictions()
        err = []
        for i in range(len(self.oob_predictions)):
            if self.oob_predictions[i] != None:
                err += [np.square(self.oob_predictions[i] - self.target[i])]
        return np.sum(err) / len(err)
    