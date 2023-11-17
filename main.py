import pickle
from tensorflow import keras

seq_data_path = {
    "x_test" : r"./Data/seq/X_test.pickle",
    "y_test" : r"./Data/seq/Y_test.pickle",
    "x_train" : r"./Data/seq/X_train.pickle",
    "y_train" : r"./Data/seq/Y_train.pickle"
}

model_json_path = r"./Models/rnn20210715-203913.json"
model_weight_path = r"./Models/rnn20210715-203913.h5"

def main():
    with open(seq_data_path["x_test"], "rb") as f:
        x_test = pickle.load(f) # (2771, 60, 195)
    with open(seq_data_path["y_test"], "rb") as f:
        y_test = pickle.load(f) # (2771, 5)

    # Load json and create model
    with open(model_json_path, 'r') as json_file:
        model_json = json_file.read()
    model = keras.models.model_from_json(model_json)

    # Load weights into new model
    model.load_weights(model_weight_path)

    # Make predictions
    predictions = model.predict(x_test) # (2771, 5)

    # Evaluate accuracy
    acc = 0
    for i in range(len(predictions)):
        if predictions[i].argmax() == y_test[i].argmax():
            acc += 1
    acc /= len(predictions)
    print("Accuracy: ", acc)

if __name__ == "__main__":
    main()