#building docker image on a Mac
version=$(awk -F'"' '/version=/ {print $2}' setup.py)

docker pull --platform linux/amd64 brunoasm/varkoder:latest

docker buildx build --push \
  --platform linux/amd64 \
  --cache-from brunoasm/varkoder:latest \
  -t brunoasm/varkoder:$version \
  -t brunoasm/varkoder:latest .

