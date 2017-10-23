import io
import argparse
from collections import defaultdict

SOS = '<s>'
EOS = '</s>'

def train_hmm(training_file, model_file):
    """The training algorithm for HMM, described in Neubig's slide p.9.
    """
    emit = defaultdict(int)
    transition = defaultdict(int)
    context = defaultdict(int)
    with open(training_file, 'r') as f:
        for line in f:
            previous = SOS  # Make the sentence start.
            context[previous] += 1
            for wordtag in line.strip().split(' '):
                word, tag = wordtag.split('_')
                # Count the transition.
                transition['{} {}'.format(previous, tag)] += 1
                context[tag] += 1  # Count the context.
                # Count the emission.
                emit['{} {}'.format(tag, word)] += 1
                previous = tag
            # Make the sentence end.
            transition['{} {}'.format(previous, EOS)] += 1

    # Save the info into a buffer temporarily.
    out = io.StringIO()
    for key, value in sorted(transition.items(),
                             key=lambda x: x[1], reverse=True):
        previous, word = key.split(' ')
        out.write('T {} {}\n'.format(key, value / context[previous]))
    for key, value in sorted(emit.items(),
                             key=lambda x: x[1], reverse=True):
        previous, tag = key.split(' ')
        out.write('E {} {}\n'.format(key, value / context[previous]))

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

    train_hmm(args.training_file, args.model_file)
