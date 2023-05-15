"""
TO RUN:
    python3 tsv_maker.py --input_matrix example.bin --input_word_index example.txt --output output.tsv --min_score 5 --unique False
"""

try:
    import numpy as np
except:
    print("Install numpy !")
    exit()

try:
    from scipy.sparse import coo_matrix
except:
    print("Install scipy !")
    exit()

import argparse
parser = argparse.ArgumentParser(description='Generates a tsv file')
parser.add_argument('--input_matrix', type=str, help='Path to input file/ cooccurance matrix')
parser.add_argument('--input_word_index', type=str, help='Path to input file/ cooccurance matrix')
parser.add_argument('--output', type=str, help='Path to output file/ tsv file')
parser.add_argument('--min_score', type=int, help='Minimum Score to consider before saving', default=5)
parser.add_argument('--unique', type=bool, help='Save unique pairs only (Default is False)', default=False)
args = parser.parse_args()

def create_tsv_file(cooccurrence_matrix_file, word_index_dict_file, output_file, min_score=5, unique=False):
    # Load the co-occurrence matrix
    cooccurrence_matrix = np.load(cooccurrence_matrix_file, allow_pickle=True)
    cooccurrence_matrix = cooccurrence_matrix.item()  # Convert to dictionary
    cooccurrence_matrix = coo_matrix(cooccurrence_matrix)

    # Load the word index
    with open(word_index_dict_file, 'r') as f:
        word_index_dict = {i: word.strip() for i, word in enumerate(f)}

    # Create the TSV file
    with open(output_file, 'w') as f:
        # Write the header
        f.write("word1\tword2\tscore\n")
        
        if (not unique):
            # Iterate over the co-occurrence matrix entries and write to the file
            for i, j, v in zip(cooccurrence_matrix.row, cooccurrence_matrix.col, cooccurrence_matrix.data):
                if v >= min_score:
                    word1 = word_index_dict[i]
                    word2 = word_index_dict[j]
                    line = f"{word1}\t{word2}\t{v}\n"
                    f.write(line)
        else:
            # Set to store (word1, word2) pairs already processed
            processed_pairs = set()
            
            # Iterate over the co-occurrence matrix entries and write to the file
            for i, j, v in zip(cooccurrence_matrix.row, cooccurrence_matrix.col, cooccurrence_matrix.data):
                if v >= min_score:
                    word1 = word_index_dict[i]
                    word2 = word_index_dict[j]
                    
                    # Check if (word2, word1) pair already processed
                    if (word2, word1) in processed_pairs:
                        continue  # Skip duplicated pair
                    
                    line = f"{word1}\t{word2}\t{v}\n"
                    f.write(line)
                    
                    # Add (word1, word2) and (word2, word1) to processed pairs
                    processed_pairs.add((word1, word2))
                    processed_pairs.add((word2, word1))

    print(f"Co-occurrence matrix saved as {output_file}")
    return 0

if __name__=='__main__':
    cooccurrence_matrix_file = args.input_matrix
    word_index_dict_file = args.input_word_index
    output_file = args.output
    min_score = args.min_score
    unique = args.unique
    create_tsv_file(cooccurrence_matrix_file, word_index_dict_file, output_file, min_score, unique)
