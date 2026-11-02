"""Independently download the checksum-pinned official Tectonic release on HPC."""
import hashlib,io,os,tarfile,urllib.request
from pathlib import Path
from amem_reproduction.storage import atomic_json,utc_now
assert os.environ.get('TMUX')
root=Path(__file__).resolve().parents[1]
url='https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/tectonic-0.17.0-x86_64-unknown-linux-musl.tar.gz'
with urllib.request.urlopen(url,timeout=120) as f:body=f.read()
sha=hashlib.sha256(body).hexdigest()
assert sha=='8533d07f9ccbd7a65824b9e0459041bca34af1eb33daba48f59215593753a3b7'
(root/'tools').mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(body),mode='r:gz') as archive:
    member=next(m for m in archive.getmembers() if m.isfile() and Path(m.name).name=='tectonic')
    (root/'tools/tectonic').write_bytes(archive.extractfile(member).read())
(root/'tools/tectonic').chmod(0o755)
atomic_json(root/'benchmark/report_template/compiler.json',dict(url=url,sha256=sha,installed_utc=utc_now()))
