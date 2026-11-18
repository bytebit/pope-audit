"""List TinyTeX release assets so we can fetch a working portable TeX."""

import json
import urllib.request

url = "https://api.github.com/repos/rstudio/tinytex-releases/releases/latest"
req = urllib.request.Request(url, headers={"User-Agent": "audit/1.0"})
with urllib.request.urlopen(req, timeout=60) as r:
    d = json.load(r)
print("tag:", d.get("tag_name"))
for a in d.get("assets", []):
    print(f"  {a['name']}  {a['size'] / 1e6:.1f} MB  {a['browser_download_url']}")
