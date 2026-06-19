from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from jinja2 import Environment, FileSystemLoader

from pingpoint.db import Database

DATA_DIR = Path.home() / ".pingpoint"
db = Database(DATA_DIR)

app = FastAPI(title="pingpoint")

templates_dir = Path(__file__).parent.parent / "web" / "templates"
templates_dir.mkdir(parents=True, exist_ok=True)

env = Environment(
    loader=FileSystemLoader(str(templates_dir)),
)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    tasks = db.list_tasks()

    task_data = []
    for task in tasks:
        solutions = db.list_solutions(task.id)
        task_data.append({
            "task": task,
            "solution_count": len(solutions),
            "latest_version": solutions[-1].version if solutions else 0,
        })

    template = env.get_template("index.html")
    return template.render(
        tasks=task_data,
    )


@app.get("/task/{task_id}", response_class=HTMLResponse)
async def task_detail(task_id: str, request: Request):
    task = db.load_task(task_id)
    if task is None:
        return HTMLResponse("Task not found", status_code=404)

    solutions = db.list_solutions(task_id)

    template = env.get_template("task.html")
    return template.render(
        task=task,
        solutions=solutions,
    )


def start_web(host: str = "127.0.0.1", port: int = 8910):
    import uvicorn
    print(f"pingpoint web UI at http://{host}:{port}")
    print("Press Ctrl+C to stop")
    uvicorn.run(app, host=host, port=port, log_level="info")
