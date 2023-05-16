"""

TO RUN:
    python3 poincare_embedding.py --input input.tsv --output embedding.bin --dimension 100 --epochs 50
"""


try:
    from gensim.models.poincare import PoincareModel
except:
    print("Install gensim !")
    exit(0)
try:
    import numpy as np
except:
    print("Install numpy !")
    exit(0)

import argparse
parser = argparse.ArgumentParser(description='Generates a poincare embedding')
parser.add_argument('--input', type=str, help='Path to input tsv file / (word1, word2, score)')
parser.add_argument('--output', type=str, help='Path to output bin file/ embedding file')
parser.add_argument('--dimension', type=int, help='Dimension of Embedding Matrix', default=100)
parser.add_argument('--epochs', type=int, help='Number of Epochs', default=50)
parser.add_argument('--batch_size', type=int, help='Number of examples to train on in a single batch.', default=1000)
args = parser.parse_args()

def train_poincare_embedding(data, filename, dimension=100, epochs=50, batch_size=1000):
    """
        data = [(word1, word2, score), (word1, word3, score)]
    """
    # Create a Poincaré embedding model using Gensim
    print("Preparing to Create a Poincaré embedding model using Gensim")
    model = PoincareModel(data, size=dimension, negative=2)  # Set the desired embedding dimension

    # Train the model with loss progress
    print("Training the model with loss progress")
    model.train(epochs=epochs, batch_size=batch_size, print_every=batch_size)
    
    print("Saving the Model")
    model.save(filename)
    print(f"Model {filename} saved Successfully !")


def read_input_file(filename):
    print("Reading the TSV File...")
    data = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        print("Preparing the Data...")
        for line in lines[1:]:  # Skip the header line
            line = line.strip().split('\t')
            entity1 = line[0]
            entity2 = line[1]
            # similarity = float(line[2])
            data.append((entity1, entity2))
    return data

if __name__=="__main__":
    inp = args.input
    out = args.output
    dimension = args.dimension
    epochs = args.epochs
    batch_size = args.batch_size

    input_data = read_input_file(inp)
    train_poincare_embedding(input_data, out, dimension, epochs, batch_size)