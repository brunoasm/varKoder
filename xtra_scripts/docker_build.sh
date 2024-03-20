#building docker image on a Mac
version=$(awk -F'"' '/version=/ {print $2}' setup.py)

docker buildx build --push \
  --platform linux/amd64 \
  --cache-from brunoasm/varkoder:latest \
  --cache-to type=local,dest=$TMPDIR/buildx-cache \
  -t brunoasm/varkoder:$version \
  -t brunoasm/varkoder:latest .

