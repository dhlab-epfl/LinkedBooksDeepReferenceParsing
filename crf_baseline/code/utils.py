import sys


def setPrintToFile(filename):
    stdout_original = sys.stdout
    f = open(filename, 'w')
    sys.stdout = f
    return f,stdout_original
    

def closePrintToFile(f, stdout_original):
    sys.stdout = stdout_original
    f.close()


def load_data(file):
    words = []
    tags_1 = []
    tags_2 = []
    tags_3 = []
    tags_4 = []

    word = tags1 = tags2 = tags3 = tags4 = []
    with open (file, "r") as file:
        for line in file:
            if 'DOCSTART' not in line: #Do not take the first line into consideration
                # Check if empty line
                if line in ['\n', '\r\n']:
                    # Append line
                    words.append(word)
                    tags_1.append(tags1)
                    tags_2.append(tags2)
                    tags_3.append(tags3)
                    tags_4.append(tags4)

                    # Reset
                    word = []
                    tags1 = []
                    tags2 = []
                    tags3 = []
                    tags4 = []

                else:
                    # Split the line into words, tag #1, tag #2, tag #3
                    w = line[:-1].split(" ")
                    word.append(w[0])
                    tags1.append(w[1])
                    tags2.append(w[2])
                    tags3.append(w[3])
                    tags4.append(w[4])

    return words,tags_1,tags_2,tags_3,tags_4