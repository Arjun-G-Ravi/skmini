import sys

def show_progress_bar(curr_block, total_block, bar_length=40):
    if total_block > 0: 
        curr_percent = (curr_block/ total_block) * 100
        block = int(round(bar_length * curr_block / total_block))
        progress = f"\r[{'#' * block}{'-' * (bar_length - block)}] {curr_percent:.2f}%"
        sys.stdout.write(progress)
        sys.stdout.flush()