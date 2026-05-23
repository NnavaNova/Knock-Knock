"""Runs the NLP server."""

# Unless you want to do something special with the server, you shouldn't need
# to change anything in this file.


from fastapi import FastAPI, Request
from nlp_manager import NLPManager
from threading import Lock

app = FastAPI()
manager = NLPManager()
load_lock = Lock()
load_status = {"loading": False, "error": ""}


@app.post("/nlp")
async def nlp(request: Request) -> dict[str, list[dict]]:
    """Performs NLP RAG QA tasks.

    Args:
        request: The API request. Contains a list of questions.

    Returns:
        A `dict` with a single key, `"predictions"`, mapping to a `list` of
        `str` answers, in the same order as which appears in `request`.
    """

    inputs_json = await request.json()
    instances = inputs_json.get("instances", [])
    if not instances:
        return {"predictions": []}

    if instances[0].get("poll") is not None:
        with load_lock:
            if load_status["error"]:
                return {"predictions": [{"status": "error"}]}
            if load_status["loading"] or not manager.loaded:
                return {"predictions": [{"status": "loading"}]}
        return {"predictions": [{"status": "loaded"}]}

    # Load the corpus if it hasn't been loaded yet.
    if instances[0].get("documents") is not None:
        with load_lock:
            if manager.loaded:
                return {"predictions": [{"status": "loaded"}]}
            load_status["loading"] = True
            load_status["error"] = ""
            try:
                manager.load_corpus(instances[0]["documents"])
                load_status["loading"] = False
            except Exception as exc:
                load_status["loading"] = False
                load_status["error"] = str(exc)
                return {"predictions": [{"status": "error"}]}
        return {"predictions": [{"status": "loaded"}]}

    predictions = []
    for instance in instances:

        # Reads the question from the request.
        question = instance.get("question", "")

        # Performs NLP QA and appends the result.
        answer = manager.qa_with_documents(question)
        predictions.append(answer)

    return {"predictions": predictions}


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint for your model."""
    return {"message": "health ok"}
