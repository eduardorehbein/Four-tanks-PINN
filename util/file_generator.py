import pickle


class PFileGenerator:
    def gen_train_data_file(self, np_train_u_X, np_train_u_Y, np_train_f_X, directory):
        train_data = [np_train_u_X, np_train_u_Y, np_train_f_X]
        with open(directory + '/train_data.p', 'wb') as p_file:
            pickle.dump(train_data, p_file)

    def gen_val_data_file(self, np_val_X, np_val_Y, directory):
        validation_data = [np_val_X, np_val_Y]
        with open(directory + '/validation_data.p', 'wb') as p_file:
            pickle.dump(validation_data, p_file)

    def gen_test_data_file(self, np_test_X, np_test_Y, directory):
        test_data = [np_test_X, np_test_Y]
        with open(directory + '/test_data.p', 'wb') as p_file:
            pickle.dump(test_data, p_file)

    def gen_weights_file(self, model, directory):
        weights = model.get_weights()
        with open(directory + '/weights.p', 'wb') as p_file:
            pickle.dump(weights, p_file)
