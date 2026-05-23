"""Runs the CV server."""

# Unless you want to do something special with the server, you shouldn't need
# to change anything in this file.


import base64
from typing import Any

from cv_manager import CVManager
from fastapi import FastAPI, Request

app = FastAPI()
manager = CVManager()


@app.post("/cv")
async def cv(request: Request) -> dict[str, list[list[dict[str, Any]]]]:
    """Performs CV object detection on image frames.

    Args:
        request: The API request. Contains a list of images, encoded in
            base-64.

    Returns:
        A `dict` with a single key, `"predictions"`, mapping to a `list` of
        `dict`s containing your CV model's predictions, in the same order as
        which appears in `request`. See `cv/README.md` for the expected format.
    """

    inputs_json = await request.json()

    image_items = [
        base64.b64decode(instance["b64"]) for instance in inputs_json["instances"]
    ]
    predictions = manager.cv_batch(image_items)

    return {"predictions": predictions}


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint for your model."""
    return {"message": "health ok"}
