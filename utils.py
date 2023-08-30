import csv
import emoji
import numpy as np

def read_glove_vecs(glove_file):
    """
    Read pre-trained word vectors from the given GloVe file.

    Args:
    glove_file (str): Path to the GloVe file containing word vectors.

    Returns:
    words_to_index (dict): A dictionary mapping words to their index.
    index_to_words (dict): A dictionary mapping index to words.
    word_to_vec_map (dict): A dictionary mapping words to their corresponding word vectors.
    """

    # Initialize dictionaries and variables
    words_to_index = {}
    index_to_words = {}
    word_to_vec_map = {}

    # Read and process the GloVe file
    with open(glove_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip().split()
            curr_word = line[0]
            words_to_index[curr_word] = i
            index_to_words[i] = curr_word
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

    return words_to_index, index_to_words, word_to_vec_map


def sentences_to_indices(X, word_to_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to `Embedding()` (described in Figure 4). 
    
    Arguments:
    X -- array of sentences (strings), of shape (m,)
    word_to_index -- a dictionary containing each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this. 
    
    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """
    
    m = X.shape[0]  # number of training examples
    
    # Initialize X_indices as a numpy matrix of zeros and the correct shape
    X_indices = np.zeros((m, max_len))
    
    for i in range(m):  # loop over training examples
        # Convert the ith training sentence to lower case and split it into words.
        sentence_words = X[i].lower().split()
        
        j = 0  # Initialize j to 0 for each sentence
        
        # Loop over the words of sentence_words
        for w in sentence_words:
            if w in word_to_index:
                # Set the (i,j)th entry of X_indices to the index of the correct word.
                X_indices[i, j] = word_to_index[w]
                j += 1  # Increment j to move to the next index within the same sentence
            
    return X_indices

def convert_to_one_hot(Y, C):
    """
    Convert an array of class labels into one-hot encoded vectors.
    
    Arguments:
    Y -- array of class labels, shape (m,)
    C -- number of classes
    
    Returns:
    Y_one_hot -- one-hot encoded matrix, shape (m, C)
    """
    # eye function creates an identity matrix
    Y_one_hot = np.eye(C)[Y.reshape(-1)]
    return Y_one_hot


def read_csv(filename):
    """
    Read data from a CSV file containing phrases and emoji labels.
    
    Args:
    filename (str): Path to the CSV file.
    
    Returns:
    X (numpy.ndarray): NumPy array of input phrases.
    Y (numpy.ndarray): NumPy array of emoji labels.
    """
    
    phrases = []
    emojis = []

    with open(filename, 'r') as csvDataFile:
        csvReader = csv.reader(csvDataFile)

        for row in csvReader:
            phrases.append(row[0])
            emojis.append(int(row[1]))  # Convert emoji label to integer

    X = np.asarray(phrases)
    Y = np.asarray(emojis, dtype=int)

    return X, Y

# Define emoji dictionary using Unicode codes and aliases
emoji_dictionary = {
    "0": "\u2764\uFE0F",  
    "1": ":baseball:",       
    "2": ":smile:",
    "3": ":disappointed:",
    "4": ":fork_and_knife:"
}

def label_to_emoji(label):
    """
    Converts a label (int or string) into the corresponding emoji code (string) ready to be printed.
    
    Args:
    label (int or str): The label representing the emoji code.
    
    Returns:
    str: The corresponding emoji code as a string.
    """
    return emoji.emojize(emoji_dictionary[str(label)])
