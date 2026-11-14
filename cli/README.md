# nuviz-cli

The `nuviz` command-line tool: a terminal dashboard and analysis toolkit for
experiments recorded with the [`nuviz`](https://pypi.org/project/nuviz/)
Python logger.

```bash
pip install "nuviz[cli]"      # logger + prebuilt CLI binary
# or
cargo install nuviz-cli
```

```bash
nuviz watch                   # live dashboard for the latest run
nuviz compare exp-a exp-b     # overlay runs on shared axes
nuviz ls                      # list experiments and their status
nuviz leaderboard --aggregate # rank configurations, mean ± std over seeds
```

See the [project README](https://github.com/LordKnows/nuviz#readme) and the
[usage guide](https://github.com/LordKnows/nuviz/blob/main/docs/USAGE.md)
for everything else.
