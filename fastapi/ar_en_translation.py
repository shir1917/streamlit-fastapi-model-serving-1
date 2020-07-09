import numpy as np
import pandas as pd

from transformers import MarianTokenizer, MarianMTModel
from typing import List

def ar_en_translation(text):
    src = 'ar'  # source language
    trg = 'en'  # target language
    sample_text = text
    mname = f'Helsinki-NLP/opus-mt-{src}-{trg}'

    model = MarianMTModel.from_pretrained(mname)
    tok = MarianTokenizer.from_pretrained(mname)
    batch = tok.prepare_translation_batch(src_texts=[sample_text])  # don't need tgt_text for inference
    gen = model.generate(**batch)  # for forward pass: model(**batch)
    words: List[str] = tok.batch_decode(gen, skip_special_tokens=True)
    return words
# print(words)
# ar_en_translation('احتفال محمد صلاح بعد ان فك عقدته قدام مانشستر يونايتد ❤️❤️ ')