import io
import argparse
import math
from collections import defaultdict

SOS = '<s>'
EOS = '</s>'
INF = 1e15  # A very large number.
LAMBDA_1 = 0.95
LAMBDA_UNK = 1 - LAMBDA_1
V = 1e210  # Vocabulary size. This value should be set very large for CJK langs!

def load_model(model_file):
    probs = defaultdict(float)
    with open(model_file, 'r') as f:
        for line in f:
            word, prob = line.rstrip().split('\t')
            probs[word] = float(prob)
    return probs

def forward(probs_uni, line):
    best_edge = {}
    best_score = {}
    best_edge[0] = None
    best_score[0] = 0.
    for word_end in range(1, len(line) + 1):
        best_score[word_end] = INF
        for word_begin in range(0, word_end):
            word = line[word_begin:word_end]  # Get the substring.
            prob = LAMBDA_UNK / V
            if word in probs_uni:
                prob += LAMBDA_1 * probs_uni[word]
            my_score = best_score[word_begin] + (-math.log2(prob))
            if my_score < best_score[word_end]:
                best_score[word_end] = my_score
                best_edge[word_end] = (word_begin, word_end)
    return best_edge

def backward(best_edge, line):
    words = []
    next_edge = best_edge[len(best_edge) - 1]
    while next_edge != None:
        # Add the substring for this edge to the words.
        word = line[next_edge[0]:next_edge[1]]
        words.append(word)
        next_edge = best_edge[next_edge[0]]
    words.reverse()
    return words

def word_segmentation(probs_uni, test_file, output_file):
    line_ws = []
    with open(test_file, 'r') as f:
        for line in f:
            line = line.strip()
            best_edge = forward(probs_uni, line)
            words = backward(best_edge, line)
            line_ws.append(' '.join(words))

    # Write the segmented lines into a buffer.
    out = io.StringIO()
    out.write('\n'.join(line_ws))

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

    probs_uni = load_model(args.model_file)
    word_segmentation(probs_uni, args.test_file, args.output_file)
