import io
import math
import argparse
from collections import defaultdict

EOS = '</s>'
LAMBDA_1 = 0.95
LAMBDA_UNK = 1 - LAMBDA_1
V = 1e6  # Vocabulary size.

def load_model(model_file):
    probabilities = defaultdict(float)
    with open(model_file, 'r') as f:
        for line in f:
            word, probability = line.strip().split('\t')
            probabilities[word] = float(probability)
    return probabilities

def test_unigram(probabilities, test_file):
    W = 0  # Total number of words.
    unk = 0  # Number of unknown words. (for calculating coverage.)
    H = 0  # Negative log likelihood.
    with open(test_file, 'r') as f:
        for line in f:
            words = line.strip().split(' ')
            words.append(EOS)
            for word in words:
                W += 1
                P = LAMBDA_UNK / V
                if word in probabilities:
                    P += LAMBDA_1 * probabilities[word]
                else:
                    unk += 1
                H += - math.log2(P)

    # Entropy H is average negative log2 likelihood per word.
    print('Entropy: {}'.format(H/W))
    # Coverage is the percentage of known words in the corpus.
    print('Coverage: {}'.format((W-unk)/W))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-file', type=str)
    parser.add_argument('--test-file', type=str)
    args = parser.parse_args()

    probabilities = load_model(args.model_file)
    test_unigram(probabilities, args.test_file)
