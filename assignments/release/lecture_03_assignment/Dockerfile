# Используем официальный образ Elasticsearch
FROM docker.elastic.co/elasticsearch/elasticsearch:8.10.1

# Установка параметров Elasticsearch (опционально)
# Здесь вы можете настроить различные параметры, такие как путь данных, память и другие параметры.

# Пример настройки пути данных:
# ENV path.data=/path/to/data

# Пример настройки объема памяти:
# ENV ES_JAVA_OPTS="-Xms512m -Xmx512m"

# Пример настройки пути логов:
# ENV path.logs=/path/to/logs

# Опционально: установка пароля для пользовательской аутентификации
# Замените "your_password_here" на ваш реальный пароль
# RUN bin/elasticsearch-setup-passwords auto --batch --url http://localhost:9200 --output /usr/share/elasticsearch/config/elasticsearch-setup-passwords

# Опционально: установка дополнительных плагинов
# RUN bin/elasticsearch-plugin install analysis-icu

# Опционально: копирование конфигурационных файлов
COPY elasticsearch.yml /usr/share/elasticsearch/config/elasticsearch.yml

# Опционально: установка параметров Kibana
# ENV SERVER_NAME=my-kibana
# ENV ELASTICSEARCH_HOSTS=http://elasticsearch:9200

# Опционально: копирование пользовательских скриптов и конфигураций
# COPY scripts/ /usr/share/elasticsearch/scripts/
# COPY config/ /usr/share/elasticsearch/config/

# Опционально: установка дополнительных настроек безопасности
# COPY elasticsearch.keystore /usr/share/elasticsearch/config/elasticsearch.keystore
# COPY elasticsearch.keystore.password /usr/share/elasticsearch/config/elasticsearch.keystore.password

# Опционально: установка дополнительных сертификатов (если необходимо)
# COPY certs/ /usr/share/elasticsearch/config/certs/

# Экспонировать порты Elasticsearch
EXPOSE 9200
EXPOSE 9300

# Запуск Elasticsearch внутри контейнера
CMD ["bin/elasticsearch"]






