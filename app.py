from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.graphs.graph_builder import GraphBuilder
from src.llms.groqllm import GroqLLM


app = FastAPI()


class BlogRequest(BaseModel):
    topic: str = Field(min_length=1)
    current_language: str = "English"
    usecase: str = "topic"


@app.post("/blogs")
async def create_blogs(request: BlogRequest):
    topic = request.topic.strip()

    if not topic:
        raise HTTPException(
            status_code=400,
            detail="Topic is required"
        )

    usecase = request.usecase.strip().lower()
    language = request.current_language.strip().lower()

    if usecase == "lang" and language not in ("hindi", "french"):
        raise HTTPException(
            status_code=400,
            detail="Language must be Hindi or French for the lang workflow"
        )

    try:
        groqllm = GroqLLM()
        llm = groqllm.get_llm()

        graph_builder = GraphBuilder(llm)
        graph = graph_builder.setup_graph(usecase=usecase)

        initial_state = {"topic": topic}

        if usecase == "lang":
            initial_state["current_language"] = language

        state = graph.invoke(initial_state)

        return {"data": state}

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        ) from exc