# Fast commands
run:
	docker-compose up

stop:
	docker-compose stop

build:
	docker build -t styler_bot_img .

entry:
	docker exec -it styler_bot bash
