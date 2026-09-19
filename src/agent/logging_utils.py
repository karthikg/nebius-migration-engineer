from rich.console import Console

console = Console()


def narrate(msg: str) -> str:
    """Print a decision-log line live and return it for the graph state."""
    console.print(f"[dim cyan]  › {msg}[/dim cyan]", highlight=False)
    return msg
