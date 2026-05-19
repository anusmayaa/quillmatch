import pickle
import json
from features import extract_features

with open('model.pkl', 'rb')      as f: model      = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f: vectorizer = pickle.load(f)

text = """It is a truth universally acknowledged that a single man
in possession of a good fortune must be in want of a wife. However little
known the feelings or views of such a man may be on his first entering a
neighbourhood this truth is so well fixed in the minds of the surrounding
families that he is considered the rightful property of some one or other
of their daughters. My dear Mr. Bennet said his lady to him one day have
you heard that Netherfield Park is let at last. Mr. Bennet replied that
he had not. But it is returned she for Mrs. Long has just been here and
she told me all about it. Do not you want to know who has taken it cried
his wife impatiently. You want to tell me and I have no objection to
hearing it. Why my dear you must know Mrs. Long says that Netherfield is
taken by a young man of large fortune from the north of England."""

tfidf_vector  = vectorizer.transform([text])
probs         = model.predict_proba(tfidf_vector)[0]
classes       = model.classes_
results       = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)

print("Direct model test — Austen passage:")
for author, prob in results:
    print(f"  {author:<25} {prob*100:.1f}%")