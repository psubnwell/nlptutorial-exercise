import io
import argparse

def count_token(corpus):
    token_count_dict = {}
    for line in corpus.strip().split('\n'):
        token = line.strip().split(' ')
        for t in token:
            if t in token_count_dict:
                token_count_dict[t] += 1
            else:
                token_count_dict[t] = 1
    return token_count_dict

if __name__ == '__main__':
    # Parse the arguments from bash.
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str)
    parser.add_argument('--output-file', type=str, default='stdout')
    arg = parser.parse_args()

    # Count tokens in the file.
    with open(arg.input_file, 'r') as f:
        token_count_dict = count_token(f.read())

    # Create a buffer and write the output info.
    out = io.StringIO()
    for token, count in sorted(token_count_dict.items(),
                               key=lambda x: x[1],
                               reverse=True):
        out.write('{}\t{}\n'.format(token, count))

    # Print on the screen or write in the file.
    if arg.output_file == 'stdout':
        print(out.getvalue().strip())
    else:
        with open(arg.output_file, 'w') as f:
            f.write(out.getvalue().strip())
