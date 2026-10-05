"""
This is for the API calls for difficulty.
def generatePrompt(model, targetDifficulty):
    candidatePrompts = [] # an array of prompts based on what level we are on
    scoredPrompts = [] # score to each prompt in the array

    # within this loop we will look at probabilities and with each level we will get the probability of what we want to challenge
    for candidate in candidates:
        testPrompt()



"""

import pandas as pd
import re

def tokenize(sentence):
    sentence = sentence.lower()
    words = re.findall(r'\b[a-z]+\b', sentence)
    return words

def trainModel():

    # 1. Load training data
    file = pd.read_csv("SpamDetectionTrainingData.csv")
    file.columns = ["Target", "data"]
    training_data = file.iloc[:]

        # counts
    spam_count = sum(training_data["Target"] == "spam")
    ham_count = sum(training_data["Target"] == "ham")
    training_count = len(training_data)

        # probability of spam in the training count
    p_spam = spam_count / training_count
    p_ham = ham_count / training_count

        # hashmaps
    spam_word_bank = {}
    ham_word_bank = {}

        # tokenize and store words in a hashmap
    for i, row in training_data.iterrows():
            # iter through every row and extract that word
        target = row["Target"]
        message = row["data"]

        words = tokenize(message)
            # store them into the appropriate map
        for word in words:
            if target == "spam":
                if word in spam_word_bank:
                    spam_word_bank[word] += 1
                else:
                    spam_word_bank[word] = 1
            elif target == "ham":
                if word in ham_word_bank:
                    ham_word_bank[word] += 1
                else:
                    ham_word_bank[word] = 1

    # total number of words in spam and ham
    spam_word_total = sum(spam_word_bank.values())
    ham_word_total = sum(ham_word_bank.values())

    # look in both hashmaps
    sentence_words = set(spam_word_bank.keys() | ham_word_bank.keys())

    # calculate the spam probability for an individual word
    def calcSpamProb(word, total):
        prob = (spam_word_bank.get(word, 0) + 1) / (total + len(sentence_words))
        return prob

    def calcHamProb(word, total):
        prob = (ham_word_bank.get(word, 0) + 1) / (total + len(sentence_words))
        return prob

    # calculate the spam probability for the words in the sentence
    def calcSentenceSpam(sentence):
        words = tokenize(sentence)
        baseProb = 1

        for word in words:
            baseProb *= calcSpamProb(word, spam_word_total)

        return baseProb

    def calcSentenceHam(sentence):
        words = tokenize(sentence)
        baseProb = 1

        for word in words:
            baseProb *= calcHamProb(word, ham_word_total)

        return baseProb

    # classify the prediction
    def classify(sentence):
        spam_prob = p_spam * calcSentenceSpam(sentence)
        ham_prob = p_ham * calcSentenceHam(sentence)

        if spam_prob > ham_prob:
            prediction = "spam"
        else:
            prediction = "ham"

        return spam_prob, ham_prob, prediction

# this function is to bring in a test prompt into the parameter
def testPrompt(text):
    print(text)

