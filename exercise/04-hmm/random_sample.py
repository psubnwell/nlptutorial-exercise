import random
import argparse
import numpy as np

SOS = '<s>'
EOS = '</s>'

def load_model(model_file):
    """Load the model file.

    Args:
        model_file: <str> The model file path.

    Returns:
        A transition probability dict and a emission probability dict.
    (Both are <dict of dicts>)
    """
    trans_prob = {}
    emiss_prob = {}
    with open(model_file, 'r') as f:
        for line in f:
            type, prev, next, prob = line.strip().split(' ')
            if type == 'T':  # Transition probability.
                if prev not in trans_prob:
                    trans_prob[prev] = {next: float(prob)}
                else:
                    trans_prob[prev].update({next: float(prob)})
            elif type == 'E':  # Emission probability.
                if prev not in emiss_prob:
                    emiss_prob[prev] = {next: float(prob)}
                else:
                    emiss_prob[prev].update({next: float(prob)})
    return trans_prob, emiss_prob

def norm(raw_list):
    """Normalize a list of float numbers.

    Args:
        raw_list: <list of float>

    Returns:
        A normalized list of float numbers.
    """
    return [i/sum(raw_list) for i in raw_list]

def random_sample(model_file):
    """Make a random sampling process in a HMM model.
    Initialize a random POS tag and generate a series of tags randomly
    according to the transition probability, as well as output a certain
    word randomly according to the emission probability.

    Args:
        model_file: <str> The model file path.

    Returns:
        A funny word sequence. :-)
    """
    trans_prob, emiss_prob = load_model(model_file)
    output_seq = []
    next_tag = random.sample(emiss_prob.keys(), 1)[0]  # Initialize.
    while next_tag != EOS:  # Until see the end of sentence mark.
        # Generate a output word.
        candidate_word = list(emiss_prob[next_tag].keys())
        candidate_word_prob = norm(emiss_prob[next_tag].values())
        output_word = np.random.choice(candidate_word, p=candidate_word_prob)
        output_seq.append(output_word)

        # Generate the next tag.
        candidate_tag = list(trans_prob[next_tag].keys())
        candidate_tag_prob = norm(trans_prob[next_tag].values())
        next_tag = np.random.choice(candidate_tag, p=candidate_tag_prob)
    print(' '.join(output_seq))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-file', type=str)
    args = parser.parse_args()

    random_sample(args.model_file)

