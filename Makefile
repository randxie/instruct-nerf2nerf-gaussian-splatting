CONTAINER_ID := $(shell docker ps | grep instruct-nerf2nerf | awk '{print $$1}')

download:
	wget https://huggingface.co/camenduru/gaussian-splatting/resolve/main/tandt_db.zip -P data/
	cd data && unzip tandt_db.zip

build:
	docker compose -f .devcontainer/docker-compose.yml build

start:
	docker compose -f .devcontainer/docker-compose.yml up

access:
	docker exec -it $(CONTAINER_ID) bash
