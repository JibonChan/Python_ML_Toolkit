"""
TO RUN:
    python3 cooccurance_maker.py --input it.txt --output it_2 --batch_size 2000 --window 5 --min_word 10 --min_freq 20
"""

try:
    import numpy as np
except:
    print("Install Numpy")
    exit(0)
try:
    from scipy.sparse import csr_matrix
except:
    print("Install Numpy")
    exit(0)
try:
    import multiprocessing
except:
    print("Install multiprocessing")
    exit(0)
try:
    from collections import defaultdict
except:
    print("Install collections")
    exit(0)

import argparse
parser = argparse.ArgumentParser(description='Generates a Coooccurance Matrix')
parser.add_argument('--input', help='Path to input file/ corpus(csv)')
parser.add_argument('--output', help='Path to output file/ co-occur matrix')
parser.add_argument('--window', type=int, help='Window Size')
parser.add_argument('--batch_size', type=int, help='Max number of sentences to process in a single batch')
parser.add_argument('--min_word', type=int, help='Minimum document length')
parser.add_argument('--min_freq', type=int, help='Minimum document length')

args = parser.parse_args()





def calculate_cooccurrence_matrix(corpus, window_size, word_index_dict, matrix_shape):
    cooccurrence_matrix = defaultdict(int)
    for sentence in corpus:
        for i, word in enumerate(sentence):
            start = max(0, i - window_size)
            end = min(len(sentence), i + window_size + 1)
            for j in range(start, end):
                if i != j:
                    cooccurrence_matrix[(word, sentence[j])] += 1
    rows, cols, data = [], [], []
    for (word1, word2), count in cooccurrence_matrix.items():
        rows.append(word_index_dict[word1])
        cols.append(word_index_dict[word2])
        data.append(count)
    cooccurrence_matrix = csr_matrix((data, (rows, cols)), shape=matrix_shape, dtype=np.float64)
    return cooccurrence_matrix

def generate_cooccurrence_matrix(corpus, window_size, min_freq, batch_size=1000):
    # Step 1: Create a frequency dictionary for words
    print("# Step 1: Create a frequency dictionary for words")
    word_freq = defaultdict(int)
    for sentence in corpus:
        for word in sentence:
            word_freq[word] += 1

    # Step 2: Filter out words with frequency less than MIN_FREQ
    print("# Step 2: Filter out words with frequency less than MIN_FREQ")
    corpus = [[word for word in sentence if word_freq[word] >= min_freq] for sentence in corpus]

    # Step 3: Create a set of unique words
    print("# Step 3: Create a set of unique words")
    unique_words = list(set(word for sentence in corpus for word in sentence))

    # Step 4: Create the word index dictionary and matrix shape
    print("# Step 4: Create the word index dictionary and matrix shape")
    word_index_dict = {word: index for index, word in enumerate(unique_words)}
    matrix_shape = (len(unique_words), len(unique_words))
    print(f"\tMatrix Shape: {matrix_shape}")

    # Step 5: Split corpus into smaller batches
    print("# Step 5: Split corpus into smaller batches")
    batches = [corpus[i:i + batch_size] for i in range(0, len(corpus), batch_size)]

    # Step 6: Parallelize the co-occurrence matrix calculation
    print("# Step 6: Parallelize the co-occurrence matrix calculation")
    pool = multiprocessing.Pool()
    results = []
    for batchc, batch in enumerate(batches, 1):
        print(f"\tWorking on Batch: {batchc}")
        result = pool.apply_async(calculate_cooccurrence_matrix, (batch, window_size, word_index_dict, matrix_shape))
        results.append(result)

    # Step 7: Accumulate the co-occurrence matrices from each batch
    print("# Step 7: Accumulate the co-occurrence matrices from each batch")
    cooccurrence_matrix = sum(result.get() for result in results)

    # Step 8: Convert the matrix to a compressed sparse row (CSR) format
    print("# Step 8: Convert the matrix to a compressed sparse row (CSR) format")
    cooccurrence_matrix = cooccurrence_matrix.tocsr()

    return cooccurrence_matrix, word_index_dict


def get_words(sentences: list):
    print("Tokenizing the words")
    result = []
    for sentence in sentences:
        words = sentence.split()
        result.append(words)
    return result

def readInput(file: str):
    print("Reading File...")
    with open(file, 'r') as f:
        file_data = f.readlines()
    new_data = []
    print(f"Filtering out documents having length < {min_word}")
    for text in file_data:
        if not len(text.split(' ')) <= min_word:
            new_data.append(text)
    return new_data

if __name__=='__main__':
    min_word = int(args.min_word)
    input_path = args.input
    output_path = args.output
    min_freq = args.min_freq
    # Example usage:
    # corpus = [
    #     ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog'],
    #     ['he', 'who', 'laughs', 'last', 'laughs', 'longest'],
    #     ['all', 'that', 'glitters', 'is', 'not', 'gold'],
    #     ['actions', 'speak', 'louder', 'than', 'words'],
    #     ['birds', 'of', 'a', 'feather', 'flock', 'together']
    # ]

    window_size = int(args.window)
    batch_size = int(args.batch_size)
    corpus = get_words(readInput(input_path))
    cooccurrence_matrix, word_index_dict = generate_cooccurrence_matrix(corpus, window_size, min_freq, batch_size)

    # Save as text file
    print("# Save as text file")
    np.savetxt(f'{output_path}_cooccurrence_matrix.txt', cooccurrence_matrix.toarray())
    np.savetxt(f'{output_path}_word_index_dict.txt', list(word_index_dict.keys()), fmt='%s')
    # Save as binary file
    print("# Save as binary file")
    np.save(f'{output_path}_cooccurrence_matrix.bin', cooccurrence_matrix)
    np.save(f'{output_path}_word_index_dict.npy', word_index_dict)
    
    exit(0)