from transformers import AutoTokenizer ,AutoModelForSequenceClassification
# from transformers import BartForSequenceClassification, BartTokenizer
# from torch.nn import functional as
import numpy as np
import pandas as pd
# from ar_en_translation import ar_en_translation
# from db_read_write import server_connection_mode, db_read, db_write

# tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
tokenizer = AutoTokenizer.from_pretrained("textattack/distilbert-base-uncased-MNLI")
# model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
model = AutoModelForSequenceClassification.from_pretrained("textattack/distilbert-base-uncased-MNLI")

def prob_label(premise, label):
    # run through model pre-trained on MNLI
    hypothesis = 'This text is about {}.'.format(label)
    input_ids = tokenizer.encode(premise, hypothesis, return_tensors='pt', max_length=tokenizer.max_len, truncation_strategy='only_first')

    logits = model(input_ids)[0]

    # we throw away "neutral" (dim 1) and take the probability of
    # "entailment" (2) as the probability of the label being true
    entail_contradiction_logits = logits[:,[0,2]]
    probs = entail_contradiction_logits.softmax(dim=1)
    # true_prob = probs[:,1].item() * 100
    # print(f'Probability that the label is true: {probs[:,1].item()* 100:0.2f}%')
    return probs[:,1].item()* 100

def print_similarities(sentences, labels):
    # for i in range(len(sentences)):
    similarities=[]
    sent_similar = []
    for j in range(len(labels)):
        similarities.append(prob_label(sentences, labels[j]))
    similarities = np.array(similarities)
    similarities_arg_sort = np.argsort(-similarities)
    # closest = -np.sort(-similarities)
    # print(sentences[i])
    for ind in similarities_arg_sort:
        sent_similar.append(f'label: {labels[ind]} \t Probability that the label is true: {similarities[ind]:0.2f}%')
    # high_ind = similarities_arg_sort[0]
    return (sent_similar)

def similarities_prob_sort(sentences, labels):
    all_sents_similarities=[]
    for i in range(len(sentences)):
        similarities=[]
        for j in range(len(labels)):
            similarities.append(prob_label(sentences[i], labels[j]))
        similarities = np.array(similarities)
        similarities_arg_sort = np.argsort(-similarities)
        all_sents_similarities.append(similarities_arg_sort)
    return all_sents_similarities

def print_results_CMD():
    labels = ['soccer', 'programming', 'sport', 'education', 'jewish', 'israel', 'palestine', 'islam', 'football', 'health care', 'movies']
    print_similarities(ar_en_translation('نزلت ترشيحات جوائز التوته الذهبيه لهذي السنه واللي تمنح لاسوأ اعمال هوليوود سنويا والمرشحين لجائزة اسوأ فيلم هم'), labels)
    print('\n')
    print_similarities(['Nominations for the Golden Raspberry Awards were awarded for this year, which are awarded to the worst acts of Hollywood annually, and the candidates for the award for the worst movie are'], labels)



# def write_results_threshold(sentences, labels, similarities_arg_sort, threshold):
#     for i in range(len(sentences)):
        # for j in

labels = ['soccer', 'programming', 'sport', 'education', 'jewish', 'israel', 'palestine', 'islam',
                      'football', 'health care', 'movies']
# prob_label(['asda asd', 'asdasd'], labels)
# similarities_prob_sort(['asda asd', 'asdasd'], labels)