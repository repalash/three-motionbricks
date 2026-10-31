#!/usr/bin/env bash
# Assemble the GitHub Pages site in _site/ (or $1): the landing page and worlds from site/, the built library
# from dist/, the robots and characters from assets/, and the plain example. Run `npm run build` first.
# The model packs are not copied: put them in _site/model/ (tools/pack_model.py) and _site/model-lite/
# (tools/pack_lite.py), as the Pages workflow does.
set -euo pipefail
cd "$(dirname "$0")/.."
out="${1:-_site}"

mkdir -p "$out/img" "$out/plain"
cp -R site/. "$out/"
cp -R dist assets "$out/"
# city/app.js bundles the library, which starts its worker from next to itself.
cp dist/motionbricks.worker.js "$out/city/"
cp docs/hero.jpg docs/run.webp docs/stride-plate.jpg docs/lineup.jpg docs/bodies.jpg docs/playground.jpg docs/plain.jpg "$out/img/"

# The plain example, with the model packs next to it and the world switcher.
src=example/index.html
grep -q "params.get('model') ?? undefined" "$src" || { echo "build_site: $src no longer matches" >&2; exit 1; }
sed -e "s#params.get('model') ?? undefined#params.get('model') ?? '../model/', liteModel: params.get('liteModel') ?? '../model-lite/'#" \
    -e "s#<title>three-motionbricks</title>#<title>Plain three.js · three-motionbricks</title>#" \
    -e "s#</body>#<script src=\"../switcher.js\" data-current=\"plain\" data-phone=\"bottom\"></script>\n</body>#" \
    "$src" > "$out/plain/index.html"

touch "$out/.nojekyll"
echo "site in $out"
