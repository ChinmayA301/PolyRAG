"""polyrag CLI: ingest -> index -> ask / compare / eval / serve."""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.table import Table

from polyrag.config import settings

app = typer.Typer(no_args_is_help=True, add_completion=False,
                  help="Multi-model RAG over real AI-governance documents.")
console = Console()


def _load_store_and_embedder():
    from polyrag.index.embedder import get_embedder
    from polyrag.index.store import VectorStore

    store = VectorStore.load(settings.index_dir)
    manifest = store.manifest
    embedder = get_embedder(manifest["embedder"], manifest.get("model") or settings.embed_model,
                            settings.embed_device, dim=manifest.get("dim"))
    return store, embedder


@app.command()
def ingest(only: list[str] = typer.Option(None, help="Only these source ids")) -> None:
    """Download sources.yaml documents and extract text (PDF text layer / OCR / Crawl4AI)."""
    from polyrag.ingest.pipeline import run_ingest

    reports = run_ingest(settings.sources_file, settings.raw_dir, settings.extracted_dir,
                         only=only or None)
    table = Table(title="Ingest report")
    for col in ("source", "type", "status", "pages", "extraction breakdown"):
        table.add_column(col)
    for r in reports:
        breakdown = ", ".join(f"{k}: {v}" for k, v in r["by_extraction"].items()) or "-"
        status = r["status"] if not r["error"] else f"failed: {r['error'][:60]}"
        table.add_row(r["source_id"], r["type"], status, str(r["pages"]), breakdown)
    console.print(table)
    failed = [r for r in reports if r["status"] != "ok"]
    if failed:
        console.print(f"[yellow]{len(failed)} source(s) failed — corpus is partial; "
                      "reports above disclose exactly what's missing.[/yellow]")


@app.command()
def index() -> None:
    """Chunk extracted text, embed (CUDA/MPS/CPU auto), and build the FAISS index."""
    from polyrag.index.chunker import chunk_records
    from polyrag.index.embedder import get_embedder
    from polyrag.index.store import VectorStore
    from polyrag.ingest.pipeline import load_extracted

    records = load_extracted(settings.extracted_dir)
    if not records:
        raise typer.Exit(console.print("[red]Nothing extracted yet — run `polyrag ingest`.[/red]") or 1)
    chunks = chunk_records(records, max_chars=settings.chunk_chars,
                           overlap=settings.chunk_overlap)
    embedder = get_embedder(settings.embedder, settings.embed_model, settings.embed_device)
    device = getattr(embedder, "device", "cpu")
    console.print(f"Embedding {len(chunks)} chunks from {len({c.source_id for c in chunks})} "
                  f"sources with [bold]{embedder.name}[/bold] on [bold]{device}[/bold]…")
    store = VectorStore.build(chunks, embedder)
    store.save(settings.index_dir)
    console.print(f"[green]Index written to {settings.index_dir}[/green] "
                  f"({store.manifest['num_chunks']} chunks, dim {store.manifest['dim']})")


@app.command()
def models() -> None:
    """List registered model aliases and whether they're ready on this machine."""
    from polyrag.llm.providers import available_aliases

    table = Table(title="Model registry")
    for col in ("alias", "provider", "model id", "ready", "notes"):
        table.add_column(col)
    for m in available_aliases():
        table.add_row(m["alias"], m["provider"], m["model_id"],
                      "✓" if m["ready"] else "✗", m["description"])
    console.print(table)


@app.command()
def ask(question: str, model: str = typer.Option("gpt-oss", "--model", "-m"),
        k: int = typer.Option(None, help="Top-k chunks"),
        as_json: bool = typer.Option(False, "--json")) -> None:
    """Ask one model, grounded in the index, with citations."""
    from polyrag import rag

    store, embedder = _load_store_and_embedder()
    result = rag.ask(question, model, store, embedder, k=k)
    if as_json:
        console.print_json(json.dumps(result.to_dict()))
        return
    c = result.completion
    if c.error:
        console.print(f"[red]{model} failed:[/red] {c.error}")
        raise typer.Exit(1)
    console.print(f"\n[bold cyan]{model}[/bold cyan] ({c.model_id}, {c.latency_s:.2f}s, "
                  f"citation coverage {result.citation_coverage:.0%})\n")
    console.print(c.text)
    console.print("\n[dim]Sources:[/dim]")
    for i, hit in enumerate(result.hits, start=1):
        mark = "*" if i in result.cited_blocks else " "
        page = f" p.{hit.chunk.page}" if hit.chunk.page is not None else ""
        console.print(f" {mark}[{i}] {hit.chunk.title}{page} (score {hit.score:.3f})")


@app.command()
def compare(question: str,
            model: list[str] = typer.Option(["gpt-oss", "gpt-oss-20b"], "--model", "-m"),
            k: int = typer.Option(None, help="Top-k chunks"),
            as_json: bool = typer.Option(False, "--json")) -> None:
    """Fan the same question + same retrieved context out to several models."""
    from polyrag import rag

    store, embedder = _load_store_and_embedder()
    results = rag.compare(question, model, store, embedder, k=k)
    if as_json:
        console.print_json(json.dumps({"results": [r.to_dict() for r in results]}))
        return
    for result in results:
        c = result.completion
        console.rule(f"[bold cyan]{c.model_alias}[/bold cyan] ({c.model_id})")
        if c.error:
            console.print(f"[red]failed:[/red] {c.error}")
            continue
        console.print(f"[dim]{c.latency_s:.2f}s | completion tokens: {c.completion_tokens} "
                      f"| citation coverage {result.citation_coverage:.0%}[/dim]\n")
        console.print(c.text)
    if results:
        console.print("\n[dim]Shared sources:[/dim]")
        for i, hit in enumerate(results[0].hits, start=1):
            page = f" p.{hit.chunk.page}" if hit.chunk.page is not None else ""
            console.print(f" [{i}] {hit.chunk.title}{page} (score {hit.score:.3f})")


@app.command("eval")
def eval_cmd(model: list[str] = typer.Option(None, "--model", "-m",
                                             help="Also run generation metrics per model"),
             k: int = typer.Option(None)) -> None:
    """Retrieval hit@k on the labeled eval set; optional per-model generation stats."""
    from polyrag.evals import evaluate, load_eval_set

    store, embedder = _load_store_and_embedder()
    eval_set = load_eval_set(settings.data_dir / "eval.yaml")
    report = evaluate(eval_set, store, embedder, k=k or settings.top_k, models=model or None)
    console.print(f"\nRetrieval hit@{report.k}: [bold]{report.hit_rate:.0%}[/bold] "
                  f"({report.retrieval_hits}/{report.retrieval_total})")
    for q in report.per_question:
        mark = "[green]✓[/green]" if q["hit"] else f"[red]✗ got {q['retrieved'][0]}[/red]"
        console.print(f" {mark} {q['question'][:80]}")
    if report.per_model:
        table = Table(title="Generation (same eval questions)")
        for col in ("model", "answered", "errors", "mean citation coverage", "mean latency (s)"):
            table.add_column(col)
        for name, s in report.per_model.items():
            table.add_row(name, str(s["answered"]), str(s["errors"]),
                          str(s["mean_citation_coverage"]), str(s["mean_latency_s"]))
        console.print(table)


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8080) -> None:
    """Run the FastAPI service (same entrypoint Docker/Cloud Run uses)."""
    import uvicorn

    uvicorn.run("polyrag.api:app", host=host, port=port)


if __name__ == "__main__":
    app()
