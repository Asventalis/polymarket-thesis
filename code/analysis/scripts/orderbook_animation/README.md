# Order-book animation

Two scripts that turn the raw book stream of one token into an animated bar chart of
the resting bids and asks, with the mid price of the hour in a lower panel.

1. `replay_book.py` runs on the server (DuckDB over the hourly partitions in
   `data/derived/partitioned/{orderbook,snapshots}`) and rebuilds the book second by
   second: the latest `book_snapshot` replaces the whole book, every later
   `price_change` sets the size resting at one price (size 0 removes the level).
   It writes one JSON with a frame per second (25 levels per side) and the recorded
   best bid / best ask of every change, kept as a check: in the example below the
   rebuilt best quotes match the recorded ones in all but 39 of the 3,600 frames
   (one-tick differences at instants with several changes).
2. `render_book.py` runs locally (matplotlib + ffmpeg or pillow) and renders an MP4
   or a GIF, or a single PNG with `--still N`.  With `--full` the price axis is the
   whole 0-1 range and the levels are summed in bins of `--bin` (a bin that holds
   both sides stacks the ask on the bid); without it the axis is a window of
   `--halfwidth` around a smoothed mid at tick resolution.  Bars taller than
   `--ymax` are clipped and labelled.

The book stream carries one token per market (the one the snapshots flag as
`YES`; the other token's book is its mirror), so `--token` must be that one.

Example used for `figures/orderbook_animation/` (market "Miami Open: Carlos Alcaraz
vs Sebastian Korda", Sebastian Korda token, 22 March 2026, 20:00–21:00 UTC: about
3,000 book changes a minute, mid from 0.59 up to 0.91 and down to 0.20):

```bash
# on the server
cd /data/v.pampinella/agy_project_dev && ./run_dev.sh replay_book.py \
    --market 0xa5012b492968df6de97bba231b55d3758852192524ad48416c8e4e52bdd8bf7b \
    --token 27161096961696559413603250893066874277369741545949668314937769262155990764064 \
    --question "Miami Open: Carlos Alcaraz vs Sebastian Korda" \
    --day 2026-03-22 --warmup-hour 19 --hour 20 --levels 5000 --out /tmp/vp_book_korda.json
# locally
uv run --no-project --with matplotlib --with pillow --with numpy python render_book.py \
    vp_book_korda.json orderbook_korda_1h.mp4 --step 1 --fps 30 --full --bin 0.01 --ymax 25000 \
    --subtitle "Polymarket, Sebastian Korda token, 1-cent levels, 1 s of video = 30 s of play"
uv run --no-project --with matplotlib --with pillow --with numpy python render_book.py \
    vp_book_korda.json orderbook_korda_12min.gif --start 1320 --end 2040 --step 2 --fps 12 \
    --full --bin 0.01 --ymax 25000 --width 9.6 --height 5.4 \
    --subtitle "Polymarket, Sebastian Korda token, 1 s of video = 24 s of play"
```
