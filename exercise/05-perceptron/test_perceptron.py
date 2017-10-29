import io
import argparse
from collections import defaultdict
import train_perceptron

def load_model(model_file):
    """Load the model from file.
    """
    w = defaultdict(float)
    with open(model_file, 'r') as f:
        for line in f:
            name, value = line.strip().split('\t')
            w[name] = float(value)
    return w

def test_perceptron(model_file, test_file, output_file):
    """Predict all on the test file with a trained model.
    (Described in Neubig's slides p.12.)
    """
    w = load_model(model_file)

    # Write the result into a buffer.
    out = io.StringIO()

    with open(test_file, 'r') as f:
        for x in f:
            phi = train_perceptron.create_features(x)
            y_prime = train_perceptron.predict_one(w, phi)
            out.write('{}\t{}\n'.format(y_prime, x.strip()))

    # Print on the screen or save in the file.
    if output_file == 'stdout':
        print(out.getvalue().strip())
    else:
        with open(output_file, 'w') as f:
            f.write(out.getvalue().strip())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-file', type=str)
    parser.add_argument('--test-file', type=str)
    parser.add_argument('--output-file', type=str, default='stdout')
    args = parser.parse_args()

    test_perceptron(args.model_file, args.test_file, args.output_file)
