#building focker image on a Mac
docker buildx build --load --platform linux/amd64 --cache-from type=local,src=/tmp/buildx-cache --cache-to type=local,dest=/tmp/buildx-cache -t brunoasm/varkoder:latest .
