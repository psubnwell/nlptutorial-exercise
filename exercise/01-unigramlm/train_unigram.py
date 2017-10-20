import io
import argparse

EOS = '</s>'

def train_unigram(training_file, model_file):
    counts = {}
    total_count = 0
    with open(training_file, 'r') as f:
        for line in f:
            words = line.strip().split(' ')
            words.append(EOS)
            for word in words:
                if word in counts:
                    counts[word] += 1
                else:
                    counts[word] = 1
                total_count += 1

    probabilities = {}
    for word, count in counts.items():
        probabilities[word] = count / total_count

    # Write the info into a buffer.
    out = io.StringIO()
    for word, probability in sorted(probabilities.items(),
                             key=lambda x: x[1], reverse=True):
        out.write('{}\t{}\n'.format(word, probability))

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

    train_unigram(args.training_file, args.model_file)
