#!/usr/bin/env python3
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """
    Finds a snippet of text within a reference document to answer a question.

    :param question: str, containing the question to answer
    :param reference: str, containing the reference document from which to
    find the answer
    :return: str, containing the answer or None if no answer is found
    """
    # Load the pre-trained BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad')

    # Load the BERT model for question answering
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

    # Tokenize the input question and reference
    inputs = tokenizer.encode_plus(
        question, reference, add_special_tokens=True, return_tensors="tf")

    # Get the input IDs and attention mask
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Make predictions using the BERT model
    outputs = model([input_ids, attention_mask])
    start_logits, end_logits = outputs["start_logits"], outputs["end_logits"]

    # Find the start and end positions of the answer
    start_index = tf.argmax(start_logits, axis=1).numpy()[0]
    end_index = tf.argmax(end_logits, axis=1).numpy()[0]

    # Extract the answer from the reference document
    if start_index <= end_index:
        answer_tokens = input_ids[0, start_index:end_index + 1]
        answer = tokenizer.decode(answer_tokens)
    else:
        answer = None

    return answer
