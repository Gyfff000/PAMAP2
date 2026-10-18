"""Lock only the transitive dependencies actually needed by this project."""
from pathlib import Path
from importlib.metadata import distribution
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
root=Path(__file__).resolve().parents[1]
pending=['numpy','scipy','scikit-learn','torch','matplotlib','joblib']
done={}
while pending:
    name=canonicalize_name(pending.pop())
    if name in done:
        continue
    dist=distribution(name)
    done[name]=dist.version
    for text in dist.requires or []:
        req=Requirement(text)
        if req.marker is None or req.marker.evaluate({'extra':''}):
            pending.append(req.name)
text='# Actual project dependency closure; CPU PyTorch wheel index.\n--extra-index-url https://download.pytorch.org/whl/cpu\n'
text+='\n'.join(f'{name}=={version}' for name,version in sorted(done.items()))+'\n'
(root/'requirements-lock.txt').write_text(text,encoding='utf-8')
print(f'Locked {len(done)} packages required by this project.')
