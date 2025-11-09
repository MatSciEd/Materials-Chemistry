
# jt-jahn-teller

A pip-installable package that reproduces the *Jahn–Teller Effect* interactive tool used in the
Materials-Chemistry repository, with the same layout and behavior.

## Install (from this repo, pinned to a tag or commit)

```python
%pip install -q "git+https://github.com/MatSciEd/Materials-Chemistry@v0.1.0#subdirectory=packages/jt-jahn-teller"
```

Or pin to an exact commit SHA:

```python
%pip install -q "git+https://github.com/MatSciEd/Materials-Chemistry@<COMMIT_SHA>#subdirectory=packages/jt-jahn-teller"
```

## Use in the notebook (two lines)

```python
from jt_jahn_teller import make_dashboard
from IPython.display import display
display(make_dashboard())
```

Optional: MP4 export (same look and logic as the notebook exporter):

```python
from jt_jahn_teller import export_jt_video
export_jt_video("jt_demo.mp4", include_atomic=True, fps=30, seconds=15)
```
