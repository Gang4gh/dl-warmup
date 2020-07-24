#!/bin/bash
sed '/^Port 22/a Port 45677' /usr/etc/sshd_config | uniq > ~/.last_sshd_config
sudo cp ~/.last_sshd_config /usr/etc/sshd_config
sudo /etc/init.d/ssh restart
