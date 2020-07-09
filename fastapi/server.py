from fastapi import FastAPI, File
import tempfile
from starlette.responses import FileResponse
from segmentation import get_segmentator, get_segments
import ZSL
model = get_segmentator()

app = FastAPI(title="Zero Shot Topic Classification",
              description='''Recently, the NLP science community has begun to pay increasing attention to zero-shot and few-shot applications, such as in the paper from OpenAI introducing GPT-3. This demo shows how 🤗 Transformers can be used for zero-shot topic classification, the task of predicting a topic that the model has not been trained on.''',
              version="0.1.0",
              )


@app.get("/text_classification/{user_input}")
async def text_classification(user_input):
    labels = ['soccer', 'programming', 'sport', 'education', 'jewish', 'israel', 'palestine', 'islam', 'football',
          'health care', 'movies']
    return {'results': str(ZSL.print_similarities(user_input, labels))}

@app.get("/")
async def root():
    return {"message": "Hello World"}


