import numpy as np


def map_at_k(predictions, ground_truth, k):
    '''
    Compute Mean Average Precision at k (mAP@k) for a given list of predictions and ground truth.

    Args:
        predictions (str): A space-separated string of predicted item IDs (e.g. '65474 646731 8993 113 3099').
        ground_truth (str): A space-separated string of ground truth item IDs (e.g. '65474 8993 646731 1163 3099').
        k (int): The number of top predictions to consider for calculating the mAP at k.

    Returns:
        float: The mean average precision at k.
    '''

    # convert the input strings to lists of integers
    predictions = list(map(int, predictions.split()))
    ground_truth = list(map(int, ground_truth.split()))

    # limit the predictions to the top-k items
    predictions_at_k = predictions[:k]

    # calculate the AP for the given k
    average_precision = 0
    relevant_count = 0

    for i, pred in enumerate(predictions_at_k):
        if pred in ground_truth:
            relevant_count += 1
            # precision at each relevant item
            average_precision += relevant_count / (i + 1)

    if relevant_count == 0:
        return 0.0  # if no relevant item is found, return 0.

    # return the average precision for this prediction
    return average_precision / min(len(ground_truth), k)


def ndcg_at_k(predictions, ground_truth, k):
    '''
    Compute Normalized Discounted Cumulative Gain at k (nDCG@k) for a given list of predictions and ground truth.

    Args:
        predictions (str): A space-separated string of predicted item IDs (e.g. '65474 646731 8993 113 3099').
        ground_truth (str): A space-separated string of ground truth item IDs (e.g. '65474 8993 646731 1163 3099').
        k (int): The number of top predictions to consider for calculating the nDCG at k.

    Returns:
        float: The normalized discounted cumulative gain at k.
    '''

    # convert the input strings to lists of integers
    predictions = list(map(int, predictions.split()))
    ground_truth = list(map(int, ground_truth.split()))

    # limit the predictions to the top-k items
    predictions_at_k = predictions[:k]

    # calculate DCG at k
    dcg = 0
    for i, pred in enumerate(predictions_at_k):
        if pred in ground_truth:
            # relevance is binary: 1 if the item is relevant, 0 otherwise
            relevance = 1
            # DCG is the sum of relevance at position i, discounted by the log of the rank (i + 1)
            # i+2 because the rank starts at 1 (i+1) and log2(1) = 0 would cause division by zero
            dcg += relevance / np.log2(i + 2)

    # calculate IDCG at k (best possible DCG for this k)
    ideal_relevance = [1 if item in ground_truth else 0 for item in predictions_at_k]
    idcg = 0
    for i, rel in enumerate(sorted(ideal_relevance, reverse=True)):
        idcg += rel / np.log2(i + 2)

    # normalize DCG to get nDCG (DCG / IDCG)
    if idcg == 0:
        return 0.0  # if IDCG is 0, return 0 to avoid division by zero

    return dcg / idcg
