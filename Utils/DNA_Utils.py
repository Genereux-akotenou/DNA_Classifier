import numpy as np

class DNA:
    @staticmethod
    def one_hot_encoding(sequences, max_length=100):
        """
        One-hot encode a list of DNA sequences.

        Parameters:
        sequences (list of str): List of DNA sequences.
        max_length (int): Maximum length of the sequences. Sequences longer than this will be truncated,
                          and sequences shorter than this will be padded with 'N'.

        Returns:
        np.ndarray: A 3D numpy array of shape (num_sequences, max_length, 4) representing the one-hot encoded sequences.
        """
        # Define a dictionary to map nucleotides to integers
        nucleotide_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}

        # Initialize an empty array for the one-hot encoded sequences
        one_hot_encoded = np.zeros((len(sequences), max_length, 4), dtype=int)

        for i, sequence in enumerate(sequences):
            # Truncate or pad the sequence to the maximum length
            sequence = sequence[:max_length].ljust(max_length, 'N')
            for j, nucleotide in enumerate(sequence):
                if nucleotide in nucleotide_to_int and nucleotide != 'N':
                    one_hot_encoded[i, j, nucleotide_to_int[nucleotide]] = 1

        return one_hot_encoded