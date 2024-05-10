import nltk
from nltk.tokenize import word_tokenize
from collections import Counter

def read_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Read the text data
text_a = read_text('textA.txt')
text_b = read_text('textB.txt')

# Combine Text A and Text B
combined_text = text_a + " " + text_b

# Tokenize the corpus into words
words = word_tokenize(combined_text)

# Determine the size N of the word vocabulary
word_vocabulary_size = len(set(words))

# Tokenize Text A into words
words_text_a = word_tokenize(text_a)

# Count the number of seen bigrams in Text A
bigrams_text_a = list(nltk.bigrams(words_text_a))
seen_bigrams_count = sum(1 for bigram in bigrams_text_a if bigram in Counter(nltk.bigrams(words)))

print("Size of word vocabulary (N):", word_vocabulary_size)
print("Number of seen bigrams in Text A:", seen_bigrams_count)
