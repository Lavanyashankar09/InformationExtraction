#!/usr/bin/env python
from dataset import AsrDataset
from model import LSTM_ASR
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
from torch import nn
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    # Step 1: Extract word spellings and features from the batch
    word_spellings = [torch.tensor(sample[0]) for sample in batch]
    features = [torch.tensor(sample[1]) for sample in batch]

    # Step 2: Pad word spellings and features
    padded_word_spellings = pad_sequence(word_spellings, batch_first=True, padding_value=25)
    padded_features = pad_sequence(features, batch_first=True, padding_value=256)

    # Step 3: Calculate lengths of unpadded word spellings and features
    list_of_unpadded_word_spelling_length = [len(sample[0]) for sample in batch]
    list_of_unpadded_feature_length = [len(sample[1]) for sample in batch]

    # Step 4: Return the padded sequences and lengths
    return padded_word_spellings, padded_features, list_of_unpadded_word_spelling_length, list_of_unpadded_feature_length


def train(train_dataloader, model, CTC_loss, optimizer):
    dataset = train_dataloader.dataset.dataset
    cc = 0
    for idx, data in enumerate(train_dataloader):
        padded_word_spellings, padded_features, list_of_unpadded_word_spelling_length, list_of_unpadded_feature_length = data
        log_prob = model(padded_features)
        words_greedy, _ = greedy_decode(log_prob, dataset)
        origin_words = paddedd(padded_word_spellings, dataset)
        count_batch_greedy = compute_accuracy(words_greedy, origin_words)
        cc += count_batch_greedy
        log_prob = log_prob.transpose(0, 1)
        loss = CTC_loss(log_prob, padded_word_spellings, list_of_unpadded_feature_length, list_of_unpadded_word_spelling_length)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  
    acc_g = cc / len(train_dataloader.dataset)
    return loss, acc_g

def validate(validate_dataloader, model, CTC_loss):
    cg = 0
    ccc = 0
    dataset = validate_dataloader.dataset.dataset
    with torch.no_grad():
        for _, (padded_word_spellings, padded_features, list_of_unpadded_word_spelling_length, list_of_unpadded_feature_length) in enumerate(validate_dataloader):
            log_prob = model(padded_features)          
            words_greedy, _ = greedy_decode(log_prob, dataset)
            words_mrd, _ = ctc_decode(log_prob, dataset, list_of_unpadded_feature_length)
            origin_words = paddedd(padded_word_spellings, dataset)    
            bgg = compute_accuracy(words_greedy, origin_words)
            count_batch_mrd = compute_accuracy(words_mrd, origin_words)
            cg += bgg
            ccc += count_batch_mrd
            log_prob = log_prob.transpose(0, 1)
            loss = CTC_loss(log_prob, padded_word_spellings, list_of_unpadded_feature_length, list_of_unpadded_word_spelling_length)
        print(f"CTC: {words_mrd}")
        print(f"Greedy: {words_greedy}")
        print(f"Ground Truth labels: {origin_words}")
        acc_g = cg / len(validate_dataloader.dataset)
        acc_c = ccc / len(validate_dataloader.dataset)
    return loss, acc_g, acc_c

def test(test_dataloader, model):
    dataset = test_dataloader.dataset
    wg = []
    wc = []
    wlg = []
    wlc = []
    with torch.no_grad():
        for idx, (_, padded_features, _, list_of_unpadded_feature_length) in enumerate(test_dataloader):
            log_prob = model(padded_features)
            gg, lgg = greedy_decode(log_prob, dataset)
            ccc, loss1 = ctc_decode(log_prob, dataset, list_of_unpadded_feature_length)
            wg.extend(gg)
            wc.extend(ccc)
            wlg.extend(lgg)
            wlc.extend(loss1)
    return wg, wc, wlg, wlc

def main():

    training_set = AsrDataset(scr_file='./data/clsp.trnscr', feature_type=feature_type1, feature_file="./data/clsp.trnlbls", feature_label_file="./data/clsp.lblnames",  wav_scp='./data/clsp.trnwav', wav_dir='./data/waveforms/')
    test_set = AsrDataset(scr_file = "./data/clsp.trnscr", feature_type=feature_type1, feature_file="./data/clsp.devlbls", feature_label_file="./data/clsp.lblnames",  wav_scp='./data/clsp.devwav', wav_dir='./data/waveforms/')
    train_size = int(0.90 * len(training_set))
    validation_size = len(training_set) - train_size
    generator = torch.Generator().manual_seed(0)
    training_set, validation_set = random_split(training_set, [train_size, validation_size], generator=generator)

    train_dataloader = DataLoader(training_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    validate_dataloader = DataLoader(validation_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = LSTM_ASR(feature_type=feature_type1, input_size=40, hidden_size=256, num_layers=2, output_size=len(training_set.dataset.letter2id))

    loss_function = torch.nn.CTCLoss(blank=training_set.dataset.blank_id)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    # Training
    train_loss_list = []
    train_grd_acc_list = []
    val_loss_list = []
    val_mrd_acc_list = []
    val_grd_acc_list = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss, accuracy_greedy_train = train(train_dataloader, model, loss_function, optimizer)
        model.eval()
        val_loss, accuracy_greedy_val, accuracy_mrd_val = validate(validate_dataloader, model, loss_function)
        
        train_loss_list.append(train_loss.item())
        train_grd_acc_list.append(accuracy_greedy_train)

        val_loss_list.append(val_loss.item())
        val_grd_acc_list.append(accuracy_greedy_val)
        val_mrd_acc_list.append(accuracy_mrd_val)
        
        print(f"Epoch: {epoch+1}")
        print(f"Training Loss: {train_loss}")
        print(f"Validation Loss: {val_loss}")
        print(f"Greedy Accuracy: {accuracy_greedy_val}")
        print(f"Min Risk Accuracy: {accuracy_mrd_val}")
    
    visualize(train_loss_list, val_loss_list, train_grd_acc_list, val_grd_acc_list, val_mrd_acc_list, feature_type1)

    # testing
    model.eval()
    words_greedy, words_mrd, words_log_prob_greedy, words_ctcloss_mrd = test(test_dataloader, model)

    words_log_prob_greedy = [i.item() for i in words_log_prob_greedy]
    words_ctcloss_mrd = [i.item() for i in words_ctcloss_mrd]

    with open(f"{feature_type1}_greedy_test_result.txt", 'w') as f:
    # Write the testing results to the text file
        for i in range(len(words_greedy)):
            f.write(f"{words_greedy[i]} {words_log_prob_greedy[i]}\n")
    with open(f"{feature_type1}_ctc_test_result.txt", 'w') as f:
        # Write the testing results to the text file
        for i in range(len(words_ctcloss_mrd)):
            f.write(f"{words_mrd[i]} {words_ctcloss_mrd[i]}\n")



def greedy_decode(log_post, dataset):
    max_log_prob_per_idx, idx = torch.max(log_post, dim=2)
    max_log_prob = torch.sum(max_log_prob_per_idx, dim=1)

    ci = [None] * idx.shape[0]
    for i in range(idx.shape[0]):
        ci[i] = torch.unique_consecutive(idx[i]).numpy().tolist()

    for i, word in enumerate(ci):
        ci[i] = [id for id in word if id not in [dataset.blank_id, dataset.silence_id, dataset.pad_id, dataset.space_id]]

    words_spelling = [[dataset.id2letter[id] for id in word] for word in ci]
    words = [''.join(word) for word in words_spelling]
    return words, max_log_prob

def ctc_decode(log_post, dataset, list_of_unpadded_feature_length):
    ctc_loss = nn.CTCLoss(blank=dataset.blank_id)
    words_list = []
    min_ctcloss_list = []
    for log_prob, unpadded_feature_length in zip(log_post, list_of_unpadded_feature_length):
        min_ctcloss = float('inf')
        selected_word = None
        for word in dataset.script:
            input = log_prob.unsqueeze(0)
            input = input.transpose(0, 1)
            spelling_of_word = [dataset.silence_id] + word + [dataset.silence_id]
            spelling_of_word = [item for sublist in [[i, dataset.space_id] for i in spelling_of_word] for item in sublist][:-1]
            target = torch.tensor([spelling_of_word])
            list_unpadded_feature_length = [unpadded_feature_length]
            list_unpadded_target_length = [len(spelling_of_word)]
            loss = ctc_loss(log_prob, target, list_unpadded_feature_length, list_unpadded_target_length)       
            if loss < min_ctcloss:
                min_ctcloss = loss
                selected_word = "".join([dataset.id2letter[id] for id in word])
        words_list.append(selected_word)
        min_ctcloss_list.append(min_ctcloss)
    return words_list, min_ctcloss_list

def paddedd(padded_word_spellings, dataset):
    ori = []
    padded_word_spellings = padded_word_spellings.numpy().tolist()
    for word in padded_word_spellings:
        word = [id for id in word if id not in [dataset.blank_id, dataset.silence_id, dataset.pad_id, dataset.space_id]]
        word = [dataset.id2letter[id] for id in word]
        word = ''.join(word)
        ori.append(word)
    return ori
   
def compute_accuracy(words, od):
    count = 0
    for word, od in zip(words, od):
        if word == od:
            count += 1
    return count

def visualize(train_loss_list, val_loss_list, train_grd_acc_list, val_grd_acc_list, val_mrd_acc_list, feature_type):
    # Plot training and validation loss
    plt.plot(train_loss_list, label="Training Loss")
    plt.plot(val_loss_list, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"./{feature_type}_loss.pdf")

    # Plot training and validation accuracy for greedy search
    plt.clf()  # Clear the current figure
    plt.plot(train_grd_acc_list, label="Training Greedy Search Accuracy")
    plt.plot(val_grd_acc_list, label="Validation Greedy Search Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"./{feature_type}_accuracy.pdf")

    # Plot validation accuracy for Minimum Risk Decode
    plt.clf()  # Clear the current figure
    plt.plot(val_mrd_acc_list, label="Validation Minimum Risk Decode Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"./{feature_type}_min_riskdecode_accuracy.pdf")

if __name__ == "__main__":
    num_epochs = 100
    feature_type1="discrete"
    batch_size=16
    main()

