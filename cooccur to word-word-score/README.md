This tsv_maker.py takes the "cooccurance_output.bin.npy" and the "word_index_pair.txt" as input from the "co-occurance/cooccurance_maker.py" and generates a tsv file:

The tsv file structure is:
    word1, word2, score


Run:
    python3 tsv_maker.py --input_matrix example.bin --input_word_index example.txt --output output.tsv --min_score 5 --unique False
