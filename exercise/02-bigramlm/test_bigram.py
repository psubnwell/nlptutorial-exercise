import io
import argparse
import math
from collections import defaultdict

SOS = '<s>'
EOS = '</s>'
V = 1e6  # Vocabulary size.

def load_model(model_file):
    probs = defaultdict(float)
    with open(model_file, 'r') as f:
        for line in f:
            ngram, prob = line.strip().split('\t')
            probs[ngram] = float(prob)
    return probs

def test_bigram(probs, test_file, lambda_1, lambda_2):
    W = 0  # Total number of words.
    H = 0  # Negative log likelihood.
    with open(test_file, 'r') as f:
        for line in f:
            words = line.strip().split(' ')
            words.insert(0, SOS)
            words.append(EOS)
            for i in range(1, len(words)):
                P1 = lambda_1 * probs[words[i]] + (1 - lambda_1) / V
                P2 = lambda_2 * probs[' '.join(words[i-1:i+1])] + \
                     (1 - lambda_2) * P1
                H += - math.log2(P2)
                W += 1
    print('Entropy: {}'.format(H/W))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-file', type=str)
    parser.add_argument('--test-file', type=str)
    parser.add_argument('--lambda-1', type=float)
    parser.add_argument('--lambda-2', type=float)
    args = parser.parse_args()

    probs = load_model(args.model_file)
    test_bigram(probs, args.test_file, args.lambda_1, args.lambda_2)
