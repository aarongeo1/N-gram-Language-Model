import argparse
import math
from collections import defaultdict, Counter

def read_data(file_path):
    with open(file_path, 'r') as f:
        data = f.readlines()
    return data

def train_ngram_model(data, n):
    model = defaultdict(Counter)
    for line in data:
        tokens = ['<s>'] + line.split() + ['</s>']
        for i in range(len(tokens) - n + 1):
            history, char = tuple(tokens[i:i + n - 1]), tokens[i + n - 1]
            model[history][char] += 1
    return model

def add_one_smoothing(model, vocabulary_size):
    for history in model:
        for char in model[history]:
            model[history][char] += 1
        model[history]['<UNK>'] = 1
    return model

def calculate_perplexity(model, data, n, vocabulary_size, laplace=False):
    log_perplexity = 0
    N = 0
    for line in data:
        tokens = ['<s>'] + line.split() + ['</s>']
        for i in range(len(tokens) - n + 1):
            history, char = tuple(tokens[i:i + n - 1]), tokens[i + n - 1]
            history_count = sum(model[history].values())
    
            # Use Laplace smoothing for zero-count bigrams
            char_count = model[history].get(char, 0)
            if laplace:
                prob = (char_count + 1) / (history_count + vocabulary_size)
            else:
                prob = (char_count / history_count) if char_count > 0 else 1 / vocabulary_size
            
            log_prob = math.log(prob)
            log_perplexity += -log_prob
            N += 1

    log_perplexity = log_perplexity / float(N)
    perplexity = math.exp(log_perplexity)
    return perplexity



def main():
    parser = argparse.ArgumentParser(description='N-gram Language Model Trainer and Perplexity Calculator')
    parser.add_argument('model_type', choices=['unigram', 'bigram', 'trigram'], help='Type of n-gram model')
    parser.add_argument('training_path', help='Path to the training data')
    parser.add_argument('eval_path', help='Path to the evaluation data')
    parser.add_argument('--laplace', action='store_true', help='Apply Laplace smoothing')
    args = parser.parse_args()

    n = 1 if args.model_type == 'unigram' else 2 if args.model_type == 'bigram' else 3
    training_data = read_data(args.training_path)
    eval_data = read_data(args.eval_path)

    # Set vocabulary_size for all scenarios
    vocabulary_size = len(set(' '.join(training_data).split())) + 1  # +1 for the OOV token

    model = train_ngram_model(training_data, n)

    if args.laplace and args.model_type in ['bigram', 'trigram']:
        model = add_one_smoothing(model, vocabulary_size)

    perplexity = calculate_perplexity(model, eval_data, n, vocabulary_size, args.laplace)
    print(f'Perplexity: {perplexity}')

if __name__ == '__main__':
    main()

