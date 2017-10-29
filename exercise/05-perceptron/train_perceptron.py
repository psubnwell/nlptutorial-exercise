import io
import argparse
from collections import defaultdict

def create_features(x):
    """Creates a perceptron model.
    (Described in Neubig's slides p.14.)
    """
    phi = defaultdict(float)
    words = x.strip().split(' ')
    for word in words:
        phi['UNI:' + word] += 1
    return phi

def predict_one(w, phi):
    """Predict a single example.
    (Described in Neubig's slides p.13.)
    """
    score = 0
    for name, value in phi.items():
        if name in w:
            score += value * w[name]  # score = w * phi(x)
    if score >= 0:
        return 1
    else:
        return -1

def update_weights(w, phi, y):
    """Update the weights of perceptron.
    (Described in Neubig's slides p.18)
    """
    for name, value in phi.items():
        w[name] += value * y
    return w

def train_perceptron(input_file, model_file):
    """Online learning for perceptron.
    (Described in Neubig's slides p.17.)
    """
    w = defaultdict(float)
    with open(input_file, 'r') as f:
        for line in f:
            y, x = line.rstrip().split('\t')
            phi = create_features(x)
            y_prime = predict_one(w, phi)  # Try to classify each example.
            if y_prime != int(y):  # If make a mistake...
                w = update_weights(w, phi, int(y))  # Update the weights.

    # Save the model into a buffer temporarily.
    out = io.StringIO()
    for key, value in w.items():
        out.write('{}\t{}\n'.format(key, value))

    # Print on the screen or save in the file.
    if model_file == 'stdout':
        print(out.getvalue().strip())
    else:
        with open(model_file, 'w') as f:
            f.write(out.getvalue().strip())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-file', type=str)
    parser.add_argument('--model-file', type=str, default='stdout')
    args = parser.parse_args()

    train_perceptron(args.training_file, args.model_file)
