import sys


def setPrintToFile(filename):
    stdout_original = sys.stdout
    f = open(filename, 'w')
    sys.stdout = f
    return f,stdout_original
    

def closePrintToFile(f, stdout_original):
    sys.stdout = stdout_original
    f.close()

def load_data(filepath):
    """
        Load and return the data stored in the given path.
        The data is structured as follows: 
            Each line contains four columns separated by a single space. 
            Each word has been put on a separate line and there is an empty line after each sentence. 
            The first item on each line is a word, the second, third and fourth are tags related to the word.
        Example:
            The sentence "L. Antonielli, Iprefetti dell' Italia napoleonica, Bologna 1983." is represented in the dataset as:
                L author b-secondary b-r
                . author i-secondary i-r
                Antonielli author i-secondary i-r
                , author i-secondary i-r
                Iprefetti title i-secondary i-r
                dell title i-secondary i-r
                ’ title i-secondary i-r
                Italia title i-secondary i-r
                napoleonica title i-secondary i-r
                , title i-secondary i-r
                Bologna publicationplace i-secondary i-r
                1983 year e-secondary i-r
                . year e-secondary e-r

        :param filepath: Path to the data
        :return: Four arrays: The first one contains sentences (one array of words per sentence) and the other threes are arrays of tags.

    """

    # Arrays to return
    words = []
    tags_1 = []
    tags_2 = []
    tags_3 = []

    word = tags1 = tags2 = tags3 = []
    with open (filepath, "r") as file:
        for line in file:
            if 'DOCSTART' not in line: #Do not take the first line into consideration
                # Check if empty line
                if line in ['\n', '\r\n']:
                    # Append line
                    words.append(word)
                    tags_1.append(tags1)
                    tags_2.append(tags2)
                    tags_3.append(tags3)

                    # Reset
                    word = []
                    tags1 = []
                    tags2 = []
                    tags3 = []

                else:
                    # Split the line into words, tag #1, tag #2, tag #3
                    w = line[:-1].split(" ")
                    word.append(w[0])
                    tags1.append(w[1])
                    tags2.append(w[2])
                    tags3.append(w[3])

    return words,tags_1,tags_2,tags_3

#def load_data(file):
#    words = []
#    tags_1 = []
#    tags_2 = []
#    tags_3 = []
#    tags_4 = []
#
#    word = tags1 = tags2 = tags3 = tags4 = []
#    with open (file, "r") as file:
#        for line in file:
#            if 'DOCSTART' not in line: #Do not take the first line into consideration
#                # Check if empty line
#                if line in ['\n', '\r\n']:
#                    # Append line
#                    words.append(word)
#                    tags_1.append(tags1)
#                    tags_2.append(tags2)
#                    tags_3.append(tags3)
#                    tags_4.append(tags4)
#
#                    # Reset
#                    word = []
#                    tags1 = []
#                    tags2 = []
#                    tags3 = []
#                    tags4 = []
#
#                else:
#                    # Split the line into words, tag #1, tag #2, tag #3
#                    w = line[:-1].split(" ")
#                    word.append(w[0])
#                    tags1.append(w[1])
#                    tags2.append(w[2])
#                    tags3.append(w[3])
#                    tags4.append(w[4])
#
#    return words,tags_1,tags_2,tags_3,tags_4
