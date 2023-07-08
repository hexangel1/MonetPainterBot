# Monet Painter Bot

Telegram бот, позволяющий придавать изображениям стиль великого художника-импрессиониста Клода Моне.

## Установка и запуск проекта

* Для сборки проекта перейдите директорию с проектом и
выполните команду `make build`  
  `$ cd MonetPainterBot`
  `$ make build`
* Создайте в корневой директории проекта файл с именем `.env`.  
В этом файле необходимо определить переменные среды `TG_BOT_TOKEN` и `CHECKPOINT_PATH`.
* Для запуска сервиса выполните команду `make run`  
  `$ make run`

## Набор команд бота

* `/start`: начало работы, запуск бота
* `/stop`: завершение сеанса
* `/help`: отображение списка доступных команд и их описание
* `/transfer_style`: команда для загрузки изображения и последующей обработки нейросетью

## Сценарий работы с ботом

* Вводим команду /start для инициализации бота.
Бот ответит приветственным сообщением и предложит ознакомиться
с набором команд. Также появятся две кнопки - `transfer_style` и `help`.
* Далее нажимаем на кнопку `/transfer_style`, либо вводим соответствующую 
команду вручную, в ответ бот предложит отправить ему изображение.
* Изображение можно отправлять как в виде фотографии, так и в виде документа.
* Отправленное изображение будет передано на вход нейросети,
после обработки изображения, бот ответит стилизованным изображением.
* Перед отправкой следующей картинки необходимо 
также отправить команду `/transfer_style` или нажать на кнопку.

## Пример диалога с ботом

<p>
<img src="https://github.com/hexangel1/TransferStyleBot/blob/main/screenshots/screenshot3.png" width="200" />
<img src="https://github.com/hexangel1/TransferStyleBot/blob/main/screenshots/screenshot4.png" width="200" />
<img src="https://github.com/hexangel1/TransferStyleBot/blob/main/screenshots/screenshot5.png" width="200" />
<img src="https://github.com/hexangel1/TransferStyleBot/blob/main/screenshots/screenshot6.png" width="200" />
</p>
<p>
<img src="https://github.com/hexangel1/TransferStyleBot/blob/main/screenshots/screenshot7.png" width="200" />
<img src="https://github.com/hexangel1/TransferStyleBot/blob/main/screenshots/screenshot8.png" width="200" />
<img src="https://github.com/hexangel1/TransferStyleBot/blob/main/screenshots/screenshot9.png" width="200" />
<img src="https://github.com/hexangel1/TransferStyleBot/blob/main/screenshots/screenshot10.png" width="200" />
</p>
<p>
<img src="https://github.com/hexangel1/TransferStyleBot/blob/main/screenshots/screenshot11.png" width="200" />
<img src="https://github.com/hexangel1/TransferStyleBot/blob/main/screenshots/screenshot12.png" width="200" />
<img src="https://github.com/hexangel1/TransferStyleBot/blob/main/screenshots/screenshot13.png" width="200" />
<img src="https://github.com/hexangel1/TransferStyleBot/blob/main/screenshots/screenshot14.png" width="200" />
</p>
