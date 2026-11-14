#!/usr/bin/env bash
# Optional, needs sudo, NOT applied on the laptop yet. Turns Wi-Fi power saving off.
# Wi-Fi on the laptop is systemd-networkd through netplan; NetworkManager doesn't manage the device.
echo 'ACTION=="add", SUBSYSTEM=="net", KERNEL=="wlo1", RUN+="/usr/sbin/iw dev wlo1 set power_save off"' | sudo tee /etc/udev/rules.d/70-wifi-powersave.rules
echo 'options iwlmvm power_scheme=1' | sudo tee /etc/modprobe.d/iwlmvm-power.conf
# only matters if NetworkManager takes over:
sudo sed -i 's/wifi.powersave = 3/wifi.powersave = 2/' /etc/NetworkManager/conf.d/default-wifi-powersave-on.conf
