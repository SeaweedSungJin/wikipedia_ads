"""Tools to evaluate answers in Encyclopedic-VQA.

This module mirrors the evaluation logic from Google's "evaluation_utils.py"
used by the Echosight project. It first normalises candidate and reference
answers, checks for an exact match, and if that fails, falls back to the
BERT-based Answer Equivalence Model (BEM) hosted on TensorFlow Hub.

The evaluation follows a two-stage process:

1. **Exact Match (EM)** – compares normalised strings and supports multi-answer
   questions via intersection-over-union.
2. **BEM** – applies a BERT classifier to determine semantic equivalence,
   accepting answers with score ≥ 0.5.

Functions:
- ``preprocess_answer`` – normalise answers for robust comparison.
- ``evaluate_example`` – public helper returning the maximum score over all
  reference answers for a given question.
"""

from __future__ import annotations

import functools
import re
import string
from typing import Any, Dict, List

import numpy as np
import scipy
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text


_VOCAB_PATH = (
    "gs://cloud-tpu-checkpoints/bert/keras_bert/uncased_L-12_H-768_A-12/vocab.txt"
)
_MODEL_PATH = "https://tfhub.dev/google/answer_equivalence/bem/1"
_PUNCTUATION_CHARACTERS = string.punctuation + "‘’´`_"
_QUESTION_TYPES = ["templated", "automatic", "multi_answer", "2_hop"]
_DIGIT_MAP = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "entailment": "yes",
    "true": "yes",
    "contradiction": "no",
    "false": "no",
}
_CONTRACTIONS = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hed've": "he'd've",
    "he'dve": "he'd've",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "Id've": "I'd've",
    "I'dve": "I'd've",
    "Im": "I'm",
    "Ive": "I've",
    "isnt": "isn't",
    "itd": "it'd",
    "itd've": "it'd've",
    "it'dve": "it'd've",
    "itll": "it'll",
    "let's": "let's",
    "maam": "ma'am",
    "mightnt": "mightn't",
    "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "notve": "not've",
    "oclock": "o'clock",
    "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at",
    "shant": "shan't",
    "shed've": "she'd've",
    "she'dve": "she'd've",
    "she's": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll",
    "somebodys": "somebody's",
    "someoned": "someone'd",
    "someoned've": "someone'd've",
    "someone'dve": "someone'd've",
    "someonell": "someone'll",
    "someones": "someone's",
    "somethingd": "something'd",
    "somethingd've": "something'd've",
    "something'dve": "something'd've",
    "somethingll": "something'll",
    "thats": "that's",
    "thered": "there'd",
    "thered've": "there'd've",
    "there'dve": "there'd've",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyd've": "they'd've",
    "they'dve": "they'd've",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "twas": "'twas",
    "wasnt": "wasn't",
    "wed've": "we'd've",
    "we'dve": "we'd've",
    "weve": "we've",
    "werent": "weren't",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whod've": "who'd've",
    "who'dve": "who'd've",
    "wholl": "who'll",
    "whos": "who's",
    "whove": "who've",
    "whyll": "why'll",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've",
    "yall": "y'all",
    "yall'll": "y'all'll",
    "y'allll": "y'all'll",
    "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've",
    "youd": "you'd",
    "youd've": "you'd've",
    "you'dve": "you'd've",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}


def preprocess_answer(
    answer: str,
    punctuation_characters: str = _PUNCTUATION_CHARACTERS,
    replacement_character: str = "",
) -> str:
    """Preprocess VQA answers by normalising punctuation and spacing."""

    def remove_articles(s: str) -> str:
        return re.sub(r"\b(the answer is|a|an|the)\b", " ", s)

    def replace_punctuation(s: str) -> str:
        to_replace = set(punctuation_characters)
        return "".join(replacement_character if c in to_replace else c for c in s)

    def white_space_fix(s: str) -> str:
        return " ".join(s.split())

    def remove_llm_span_prefix(answer: str, prefix: str = "<extra_id_0> ") -> str:
        if answer.startswith(prefix):
            return answer.replace(prefix, replacement_character)
        return answer

    def standarize_digits_and_contractions(s: str) -> str:
        output = []
        tmp = s.split()
        for w in tmp:
            w = _DIGIT_MAP.get(w, w)
            w = _CONTRACTIONS.get(w, w)
            output.append(w)
        return " ".join(output)

    answer = answer.lower().replace("\n", " ").replace("\t", " ").strip()
    answer = remove_llm_span_prefix(answer)
    answer = replace_punctuation(answer)
    answer = remove_articles(answer)
    answer = standarize_digits_and_contractions(answer)
    answer = white_space_fix(answer)

    return answer


def singleanswer_exact_match(reference: str, candidate: str) -> bool:
    """Compute exact match between single reference and candidate answers."""
    preprocessed_reference = preprocess_answer(reference)
    preprocessed_candidate = preprocess_answer(candidate)
    if not preprocessed_reference:
        raise ValueError("Reference answer is empty after preprocessing.")
    return preprocessed_reference == preprocessed_candidate


def _list_intersection_over_union(target_list: List[str], prediction_list: List[str]) -> float:
    """Computes IoU for lists for multi-answer questions."""
    if not target_list:
        raise ValueError("Target list should not be empty.")
    target_set = set(target_list)
    prediction_set = set(prediction_list)
    intersection = target_set.intersection(prediction_set)
    union = target_set.union(prediction_set)
    return len(intersection) / len(union)


def multianswer_exact_match(
    reference: str, candidate: str, iou_threshold: float = 0.5
) -> bool:
    """Computes an exact match score for multi_answer questions."""
    reference_list = reference.split("&&")
    reference_list = [preprocess_answer(a) for a in reference_list]
    reference_list = [a for a in reference_list if a]
    if not reference_list:
        raise ValueError("Reference list is empty after preprocessing.")
    candidate_list = candidate.replace(" and ", ",").replace(" & ", ",").split(",")
    candidate_list = [preprocess_answer(a) for a in candidate_list]
    candidate_list = [a for a in candidate_list if a]
    iou = _list_intersection_over_union(reference_list, candidate_list)
    return iou >= iou_threshold


def exact_match_scoring_function(example: Dict[str, Any]) -> bool:
    """Score an example using exact match (EM)."""
    if example["question_type"] == "multi_answer":
        return multianswer_exact_match(example["reference"], example["candidate"])
    return singleanswer_exact_match(example["reference"], example["candidate"])


def initialize_bem_scoring_function(
    vocab_path: str = _VOCAB_PATH, model_path: str = _MODEL_PATH
):
    """Instantiates and returns a function to compute BEM scores."""

    vocab_table = tf.lookup.StaticVocabularyTable(
        tf.lookup.TextFileInitializer(
            filename=vocab_path,
            key_dtype=tf.string,
            key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
            value_dtype=tf.int64,
            value_index=tf.lookup.TextFileIndex.LINE_NUMBER,
        ),
        num_oov_buckets=1,
    )
    cls_id, sep_id = vocab_table.lookup(tf.convert_to_tensor(["[CLS]", "[SEP]"]))
    tokenizer = text.BertTokenizer(
        vocab_lookup_table=vocab_table,
        token_out_type=tf.int64,
        preserve_unused_token=True,
        lower_case=True,
    )

    bem = hub.load(model_path)

    def preprocess_example(example: Dict[str, str]) -> Dict[str, np.ndarray]:
        question = tokenizer.tokenize(example["question"]).merge_dims(1, 2)
        reference = tokenizer.tokenize(example["reference"]).merge_dims(1, 2)
        candidate = tokenizer.tokenize(example["candidate"]).merge_dims(1, 2)

        input_ids, segment_ids = text.combine_segments(
            (candidate, reference, question), cls_id, sep_id
        )

        return {
            "input_ids": input_ids.numpy(),
            "segment_ids": segment_ids.numpy(),
        }

    def pad(a: np.ndarray, length: int = 512) -> np.ndarray:
        return np.append(a, np.zeros(length - a.shape[-1], np.int32))

    def bertify_examples(examples: List[Dict[str, str]]) -> Dict[str, np.ndarray]:
        input_ids = []
        segment_ids = []
        for example in examples:
            example_inputs = preprocess_example(example)
            input_ids.append(pad(example_inputs["input_ids"]))
            segment_ids.append(pad(example_inputs["segment_ids"]))

        return {
            "input_ids": np.stack(input_ids),
            "segment_ids": np.stack(segment_ids),
        }

    def score_example(
        example: Dict[str, str], threshold_score: bool = True
    ) -> float:
        if not example["reference"]:
            raise ValueError("Reference answer cannot be empty.")

        if example["question_type"] in ["list", "multianswer", "multi_answer"]:
            example["reference"] = example["reference"].replace("&&", ",")

        inputs = bertify_examples([example])
        logits = bem(inputs)
        score = float(scipy.special.softmax(np.squeeze(logits))[1])
        if threshold_score:
            return float(score >= 0.5)
        return score

    return score_example


def encyclopedic_vqa_evaluation_function(
    example: Dict[str, str], bem_scoring_function
) -> float:
    """Scores an example using the Encyclopedic-VQA evaluation function."""
    if not example["reference"]:
        raise ValueError("Reference answer cannot be empty.")
    if example["question_type"] not in _QUESTION_TYPES:
        raise ValueError(
            f"Unknown question type. Valid options are {_QUESTION_TYPES}"
        )
    matches_exactly = exact_match_scoring_function(example)
    if matches_exactly:
        return 1.0
    return bem_scoring_function(example, threshold_score=True)


@functools.cache
def initialize_encyclopedic_vqa_evaluation_function(
    vocab_path: str = _VOCAB_PATH, model_path: str = _MODEL_PATH
):
    """Instantiates and returns a function to compute Encyclopedic-VQA scores."""
    bem_scoring_function = initialize_bem_scoring_function(
        vocab_path=vocab_path, model_path=model_path
    )
    return functools.partial(
        encyclopedic_vqa_evaluation_function,
        bem_scoring_function=bem_scoring_function,
    )


def evaluate_example(
    question: str, reference_list: List[str], candidate: str, question_type: str
) -> float:
    """Evaluate a candidate answer against references using EVQA metric."""

    if not reference_list:
        raise ValueError("Reference list cannot be empty.")

    scoring_function = initialize_encyclopedic_vqa_evaluation_function()

    scores = []
    for reference in reference_list:
        example = {
            "question": question,
            "reference": reference,
            "candidate": candidate,
            "question_type": question_type,
        }
        score = scoring_function(example)
        scores.append(score)
    return max(scores)