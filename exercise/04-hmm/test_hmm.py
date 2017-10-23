import io
import argparse
import math
from collections import defaultdict

SOS = '<s>'
EOS = '</s>'
N = 1e6
LAMBDA = 0.95

def load_model(model_file):
    """Load the model from the file.

    Args:
        model_file: <str> The file path of the model trained by `train_hmm.py`.

    Returns:
        A tuple contains three dicts: transition, emission and possible_tags.
    """
    transition = defaultdict(float)
    emission = defaultdict(float)
    possible_tags = defaultdict(float)
    with open(model_file, 'r') as f:
        for line in f:
            type, context, word, prob = line.strip().split(' ')
            possible_tags[context] = 1
            if type == 'T':
                transition[' '.join([context, word])] = float(prob)
            else:
                emission[' '.join([context, word])] = float(prob)
    # The derived possible tags contains SOS.
    # The pseudo-code in the slides don't remove it.
    # possible_tags.pop(SOS)  # Un-comment this line if needed.
    return transition, emission, possible_tags

def prob_trans(key, model):
    """Get the transition probability from the HMM model,
    described in Neubig's slide p.10.

    Args:
        key: <str> The key of transition dict,
             usually in the form of 'WORD_{i-1} WORD_i'.
        model: <dict> The transition part of HMM model.

    Returns:
        A corresponding transition probability.
    """
    return model[key]

def prob_emiss(key, model):
    """Get the emission probability from the HMM model,
    described in Neubig's slide p.10.
    Notice that we should smooth for unknown words.

    Args:
        key: <str> The key of emission dict,
             usually in the form of 'TAG_i WORD_i'.
        model: <dict> The emission part of HMM model.

    Returns:
        A corresponding smoothed emission probability.
    """
    return LAMBDA * model[key] + (1 - LAMBDA) * 1 / N

def forward_neubig(transition, emission, possible_tags, line):
    """The forward process of the Viterbi algorithm,
    described in Neubig's slides p.42.

    Notice: this version of `forward()` is totally the same as the pseudo-code
    described in the p.42. It works but I think two points are confusing.
    1) The `possible_tags` contains SOS (default <s>).
    2) Exchanging the outer loop and the inner loop will be more fit with the
    process described in the slides.
    Refer to `forward()`.

    Args:
        transition: <dict>
        emission: <dict>
        possible_tags: <dict>
        line: <str> A line of the file.

    Returns:
        The best edges <dict> derived from the forward process.
    """
    words = line.strip().split(' ')
    l = len(words)
    best_score = {}
    best_edge = {}
    best_score['{} {}'.format(0, SOS)] = 0 # Start with SOS (default <s>).
    best_edge['{} {}'.format(0, SOS)] = None

    # First and middle part.
    for i in range(0, l):
        for prev in possible_tags.keys():
            for next in possible_tags.keys():
                prev_key = '{} {}'.format(i, prev)
                next_key = '{} {}'.format(i + 1, next)
                trans_key = '{} {}'.format(prev, next)
                emiss_key = '{} {}'.format(next, words[i])
                if prev_key in best_score and trans_key in transition:
                    score = best_score[prev_key] + \
                            -math.log2(prob_trans(trans_key, transition)) + \
                            -math.log2(prob_emiss(emiss_key, emission))
                    if next_key not in best_score or best_score[next_key] > score:
                        best_score[next_key] = score
                        best_edge[next_key] = prev_key

    # Final part.
    for prev in possible_tags.keys():
        for next in [EOS]:
            prev_key = '{} {}'.format(l, prev)
            next_key = '{} {}'.format(l + 1, next)
            trans_key = '{} {}'.format(prev, next)
            emiss_key = '{} {}'.format(next, EOS)
            if prev_key in best_score and trans_key in transition:
                score = best_score[prev_key] + \
                        -math.log2(prob_trans(trans_key, transition))
                if next_key not in best_score or best_score[next_key] > score:
                    best_score[next_key] = score
                    best_edge[next_key] = prev_key

    return best_edge

def forward(transition, emission, possible_tags, line):
    """The forward process of the Viterbi algorithm,
    described in Neubig's slides p.38-40.
    Notice: Maybe this version of `forward()` is more easy for understanding.

    Args:
        transition: <dict>
        emission: <dict>
        possible_tags: <dict>
        line: <str> A line of the file.

    Returns:
        The best edges <dict> derived from the forward process.
    """
    # Remove the SOS (default <s>) from the possible tags.
    if SOS in possible_tags:
        possible_tags.pop(SOS)
    words = line.strip().split(' ')
    l = len(words)
    best_score = {}
    best_edge = {}
    best_score['{} {}'.format(0, SOS)] = 0 # Start with SOS (default <s>).
    best_edge['{} {}'.format(0, SOS)] = None

    # Following three parts are corresponding to the Neubig's slides p.38-40.
    # I make them looks nearly the same in the forms to let you easy to compare.
    # Of course you can combine these parts together to make the code shorter!

    # First part, described in Neubig's slides p.38.
    for next in possible_tags.keys():
        for prev in [SOS]:
            prev_key = '{} {}'.format(0, prev)
            next_key = '{} {}'.format(1, next)
            trans_key = '{} {}'.format(prev, next)
            emiss_key = '{} {}'.format(next, words[0])
            if prev_key in best_score and trans_key in transition:
                score = best_score[prev_key] + \
                        -math.log2(prob_trans(trans_key, transition)) + \
                        -math.log2(prob_emiss(emiss_key, emission))
                if next_key not in best_score or best_score[next_key] > score:
                    best_score[next_key] = score
                    best_edge[next_key] = prev_key

    # Middle part, described in Neubig's slides p.39.
    for i in range(1, l):
        for next in possible_tags.keys():
            for prev in possible_tags.keys():
                prev_key = '{} {}'.format(i, prev)
                next_key = '{} {}'.format(i + 1, next)
                trans_key = '{} {}'.format(prev, next)
                emiss_key = '{} {}'.format(next, words[i])
                if prev_key in best_score and trans_key in transition:
                    score = best_score[prev_key] + \
                            -math.log2(prob_trans(trans_key, transition)) + \
                            -math.log2(prob_emiss(emiss_key, emission))
                    if next_key not in best_score or best_score[next_key] > score:
                        best_score[next_key] = score
                        best_edge[next_key] = prev_key

    # Final part, described in Neubig's slides p.40.
    for next in [EOS]:
        for prev in possible_tags.keys():
            prev_key = '{} {}'.format(l, prev)
            next_key = '{} {}'.format(l + 1, next)
            trans_key = '{} {}'.format(prev, next)
            emiss_key = '{} {}'.format(next, EOS)
            if prev_key in best_score and trans_key in transition:
                score = best_score[prev_key] + \
                        -math.log2(prob_trans(trans_key, transition))
                if next_key not in best_score or best_score[next_key] > score:
                    best_score[next_key] = score
                    best_edge[next_key] = prev_key

    return best_edge

def backward(best_edge, line):
    """The backward part of Viterbi algorithm.

    Args:
        best_edge: <list> Each component contains previous best edge.
        line: <str> A line of the file.

    Returns:
        The tag sequence.
    """
    words = line.strip().split(' ')
    l = len(words)
    tags = []
    next_edge = best_edge['{} {}'.format(l+1, EOS)]
    while next_edge != '{} {}'.format(0, SOS):
        position, tag = next_edge.split(' ')
        tags.append(tag)
        next_edge = best_edge[next_edge]
    tags.reverse()
    return tags

def test_hmm(model_file, test_file, output_file):
    transition, emission, possible_tags = load_model(model_file)

    # Write the pos tags into a buffer.
    out = io.StringIO()

    with open(test_file, 'r') as f:
        for line in f:
            best_edge = forward(transition, emission, possible_tags, line)
            tags = backward(best_edge, line)
            out.write(' '.join(tags) + '\n')

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

    test_hmm(args.model_file, args.test_file, args.output_file)
