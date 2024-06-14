import numpy as np
import itertools
import pandas as pd
from sklearn.feature_extraction import DictVectorizer

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

    @staticmethod
    def kmer_count(sequence, k=3, step=1):
        """
        Utils: to count kmer occurence in DNA sequence and compute frequence
        """
        kmers = [''.join(p) for p in itertools.product('ACGT', repeat=k)]
        kmers_count = {kmer: 0 for kmer in kmers}
        s = 0
        for i in range(0, len(sequence) - k + 1, step):
            kmer = sequence[i:i + k]
            s += 1
            #kmers_count[kmer] += 1
            if kmer in kmers_count:
                kmers_count[kmer] += 1
            #else:
            #    kmers_count[kmer] = 1
        for key, value in kmers_count.items():
            kmers_count[key] = value / s
    
        return kmers_count

    @staticmethod
    def read_fasta_file(file_path, family):
        """
        Utils: Convert fasta file to dataframe
        """
        sequences = []
        with open(file_path, 'r') as file:
            current_id = None
            current_sequence = ''
            for line in file:
                if line.startswith('>'):
                    if current_id:
                        sequences.append({'id': current_id, 'sequence':current_sequence, 'length':len(current_sequence), 'class': family})
                    current_id = line.strip().split('|')[0][1:].strip()
                    current_sequence = ''
                else:
                    current_sequence += line.strip()
            if current_id:
                sequences.append({'id': current_id, 'sequence':current_sequence, 'length':len(current_sequence), 'class': family})
        
        df = pd.DataFrame(sequences)
        return df

    @staticmethod
    def build_kmer_representation(df, k=3):
        """
        Utils: For given k-mer generate dataset and return vectorised version
        """
        sequences   = df['sequence']
        kmers_count = []
        for i in range(len(sequences)):
            kmers_count.append(DNA.kmer_count(sequences[i], k=k, step=1))
            
        v = DictVectorizer(sparse=False)
        feature_values = v.fit_transform(kmers_count)
        feature_names = v.get_feature_names_out()
        X = pd.DataFrame(feature_values, columns=feature_names)
        y = df['class']
        return X, y