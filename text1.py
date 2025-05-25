import nltk
import heapq
import re
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')

def summarize_text(text, num_sentences=3):
    # Clean and split text into sentences
    text = re.sub(r'\s+', ' ', text)
    sentences = nltk.sent_tokenize(text)

    if len(sentences) <= num_sentences:
        return text

    # Vectorize the sentences using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    sentence_vectors = vectorizer.fit_transform(sentences)

    # Compute sentence scores
    sentence_scores = sentence_vectors.sum(axis=1)
    sentence_scores = [score[0, 0] for score in sentence_scores]

    # Rank sentences by score and select the top ones
    ranked_sentences = [sentences[i] for i in heapq.nlargest(num_sentences, range(len(sentence_scores)), key=lambda i: sentence_scores[i])]

    return ' '.join(ranked_sentences)

# Example usage
text = """
Artificial Intelligence (AI) is transforming the way we live and work. From voice assistants to medical diagnosis, AI plays a crucial role in today’s technology. 
It helps businesses automate processes, make better decisions, and improve customer experiences. However, with great power comes great responsibility. 
There are also concerns about privacy, job displacement, and the ethical use of AI. Thus, it is important to balance innovation with accountability.
"""

summary = summarize_text(text, num_sentences=2)
print("Summary:\n", summary)
