FROM mongo:4.0

# Install cron
RUN apt-get update && apt-get install -y cron

# Copy crontab and scripts
COPY ./crontab-backup /etc/cron.d/crontab-backup
COPY ./backup-full.sh /backup-full.sh
COPY ./backup-mini.sh /backup-mini.sh
COPY ./backup-latest-bson.sh /backup-latest-bson.sh

# Set permissions on crontab
RUN chmod 644 /etc/cron.d/crontab-backup

# Run cron on container startup
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["cron -f"]
