import re
import numpy as np

def read_text_from_file(filename):
    with open(filename, 'r') as file:
        text = file.read()
    return text

def calculate_relative_frequency(text, Y):
    letter_counts = {letter: text.count(letter) for letter in Y}
    total_characters = sum(letter_counts.values())
    relative_frequencies = {letter: count / total_characters for letter, count in letter_counts.items()}
    return relative_frequencies

Y_size = 27
Y = "abcdefghijklmnopqrstuvwxyz "
filename = "textA.txt"
text_A = read_text_from_file(filename)
relative_frequency = calculate_relative_frequency(text_A, Y)
r_y = np.random.rand(Y_size)
mean_r_y = np.mean(r_y)
delta_y = r_y - mean_r_y
lambda_value = 0.003

while lambda_value > 0.001 :
    q_y_minus = {letter: relative_frequency[letter] - lambda_value * delta_y[i] for i, letter in enumerate(Y)}
    q_y_plus = {letter: relative_frequency[letter] + lambda_value * delta_y[i] for i, letter in enumerate(Y)}

    if all(q > 0 for q in q_y_minus.values()) and all(q > 0 for q in q_y_plus.values()):
        break
    else:
        lambda_value -= 0.00001

print("Chosen Lambda:", lambda_value)
print("\nq values with lambda for q(y|1):")
for letter, q_value in q_y_minus.items():
    print(f"Letter '{letter}': {q_value:.4f}")
print("\nq values with lambda for q(y|2):")
for letter, q_value in q_y_plus.items():
    print(f"Letter '{letter}': {q_value:.4f}")
