# from keras_preprocessing.text import Tokenizer
#
# tokenizer = Tokenizer()
# lines = ['this is good', 'that is a cat']
# tokenizer.fit_on_texts(lines)
#
# res = tokenizer.texts_to_sequences(['cat is good'])
# print(res[0])
#

from nltk.translate.bleu_score import corpus_bleu

references = [[['there', 'is', 'a', 'cat', 'and', 'a', 'dog']]]
candidate = [['there', 'is', 'a', 'cat', 'and', 'a', 'pig']]

score = corpus_bleu(references, candidate, weights=(0.25, 0.25, 0.25, 0.25))
print(score)