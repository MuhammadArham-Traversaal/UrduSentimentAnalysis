INFER_PROMPT_TEMPLATE = """You are an expert in Urdu linguistics and sentiment analysis, with deep understanding of cultural context and linguistic nuances in Urdu language. Your task is to classify Urdu text reviews as strictly Positive or Negative.
Instructions:

Read the provided Urdu review carefully and classify the review sentiment as either Postive or Negative considering the sentiment within the cultural context, sarcasm, and overall tone of the review.

## Classification:
Output only a single label from one of the following options:
1. Positive: Reviews expressing satisfaction, approval, praise, or recommendation
2. Negative: Reviews expressing dissatisfaction, criticism, disapproval, or complaints


## Classify Input Sentiment. Strictly output only Positive or Negative label keyword:
Review: {text}
Sentiment Classification: """

SPLIT_ON_TERM = "Sentiment Classification: "
