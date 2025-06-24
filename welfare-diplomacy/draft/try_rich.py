from rich.progress import Progress
import time

with Progress() as progress:

    task1 = progress.add_task("[red]Scraping", total=100)

    while not progress.finished:
        progress.update(task1, advance=0.5)
        time.sleep(0.5)