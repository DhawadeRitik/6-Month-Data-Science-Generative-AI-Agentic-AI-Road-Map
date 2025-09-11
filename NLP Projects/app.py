from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import re
import pandas as pd
import pickle
from io import BytesIO
import base64
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load NLTK stopwords
STOPWORDS = set(stopwords.words("english"))

# Load FastAPI
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load model & vectorizer once at startup
predictor = pickle.load(open(r"best_rf.pkl", "rb"))
vectorizer = pickle.load(open(r"tfidf_vectorizer.pkl", "rb"))


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("landing.html", {"request": request})


@app.post("/predict")
async def predict(text: str = Form(None), file: UploadFile = File(None)):
    try:
        if file:
            # Bulk prediction from CSV
            df = pd.read_csv(file.file)
            df, graph_b64 = bulk_prediction(df)

            # Return CSV + graph as headers
            csv_bytes = df.to_csv(index=False).encode()
            headers = {
                "X-Graph-Exists": "true",
                "X-Graph-Data": graph_b64
            }
            return StreamingResponse(
                BytesIO(csv_bytes),
                media_type="text/csv",
                headers=headers
            )

        elif text:
            # Single prediction
            result = single_prediction(text)
            return JSONResponse({"prediction": result})

        else:
            return JSONResponse({"error": "No input provided"}, status_code=400)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------- Helper functions ----------

def clean_text(text_input: str):
    """Preprocess text just like during training"""
    stemmer = PorterStemmer()
    review = re.sub("[^a-zA-Z]", " ", text_input)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
    return " ".join(review)


def single_prediction(text_input: str):
    review = clean_text(text_input)
    X_prediction = vectorizer.transform([review]).toarray()
    y_pred = predictor.predict(X_prediction)[0]
    return sentiment_mapping(y_pred)


def bulk_prediction(data: pd.DataFrame):
    corpus = [clean_text(str(sentence)) for sentence in data["Sentence"]]
    X_prediction = vectorizer.transform(corpus).toarray()
    y_predictions = predictor.predict(X_prediction)

    y_predictions = list(map(sentiment_mapping, y_predictions))
    data["Predicted sentiment"] = y_predictions

    graph_b64 = get_distribution_graph(data)
    return data, graph_b64


def get_distribution_graph(data):
    fig = plt.figure(figsize=(5, 5))
    colors = ("green", "red")
    wp = {"linewidth": 1, "edgecolor": "black"}
    tags = data["Predicted sentiment"].value_counts()
    explode = (0.01, 0.01)

    tags.plot(
        kind="pie",
        autopct="%1.1f%%",
        shadow=True,
        colors=colors,
        startangle=90,
        wedgeprops=wp,
        explode=explode,
        title="Sentiment Distribution",
        xlabel="",
        ylabel="",
    )

    graph = BytesIO()
    plt.savefig(graph, format="png")
    plt.close()

    return base64.b64encode(graph.getbuffer()).decode("ascii")


def sentiment_mapping(x):
    return "Positive" if x == 1 else "Negative"
