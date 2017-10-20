import io
import argparse
from collections import defaultdict

SOS = '<s>'
EOS = '</s>'

def train_bigram(training_file, model_file):
    counts = defaultdict(int)
    context_counts = defaultdict(int)
    with open(training_file, 'r') as f:
        for line in f:
            token = line.strip().split(' ')
            token.insert(0, SOS)
            token.append(EOS)
            for i in range(1, len(token)): # starting at 1, after <s>
                # Add bigram and bigram context.
                counts[' '.join(token[i-1:i+1])] += 1  # Number of 'w_i, w_{i-1}'
                context_counts[token[i-1]] += 1  # Number of w_{i-1}.
                # Add unigram and unigram context.
                counts[token[i]] += 1  # Number of w_i.
                context_counts[''] += 1  # Total number of words.

    probabilities = {}
    for ngram, count in sorted(counts.items(),
                               key=lambda x:x[1],
                               reverse=True):
        # Notice the lines below are compatible with n-gram too!
        words = ngram.split(' ')
        context = words[:-1]
        context = ' '.join(context)
        probabilities[ngram] = count / context_counts[context]

    # Write the model's content into a buffer.
    out = io.StringIO()
    for ngram, probability in sorted(probabilities.items(),
                                     key=lambda x: x[1], reverse=True):
        out.write('{}\t{}\n'.format(ngram, probability))

    # Print on the screen or save in the model file.
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

    train_bigram(args.training_file, args.model_file)
