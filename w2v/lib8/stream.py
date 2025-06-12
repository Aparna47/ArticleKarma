
def stream_sentences(file_path):
    """
    A generator that reads a file and yields sentences as lists of words.
    This is a memory-efficient replacement for gensim.models.word2vec.LineSentence.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            yield line.split()
            