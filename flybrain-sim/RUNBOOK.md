# Fly Brain — Quick Start

## Every run

1. Open WSL terminal, start SSH and the sim:

    sudo service ssh start
    source /home/mrmotts/fly-brain/doomfly/.venv-neural/bin/activate
    cd /home/mrmotts/fly-brain/flybrain-sim-repo/flybrain-sim
    python3 brain_runner.py

   Defaults: 1600x1200 arena, 4 obstacles. No flags needed.

2. Open in Chrome (Windows):

    \\wsl.localhost\Ubuntu\home\mrmotts\fly-brain\flybrain-sim-repo\flybrain-sim\fly.html

## After a reboot (if SSH times out)

WSL2 IP may have changed. In WSL terminal:

    ip addr show eth0 | grep 'inet ' | awk '{print $2}' | cut -d/ -f1

If IP changed from 172.30.131.89, update Windows port proxy (PowerShell admin):

    netsh interface portproxy delete v4tov4 listenport=2222 listenaddress=0.0.0.0
    netsh interface portproxy add v4tov4 listenport=2222 listenaddress=0.0.0.0 connectport=22 connectaddress=NEW_IP

## HUD — turn sources

  dn (blue)        DNa02 — optic flow steering
  loom (orange)    DNp01 — looming escape
  p04 (yellow)     DNp04 — smooth avoidance
  contact (purple) DNg29 — touch escape
  scatter (red)    random kick (fires briefly on escape)

Open field: dn flickers, rest near zero.
Near wall/obstacle: loom + p04 light up before turn; scatter spikes on kicks.

## Useful flags

  --fps 30        reduce WS rate if browser is choppy
  --no-looming    ablate looming pathway
  --no-scatter    ablate scatter kicks
  --obstacles 0   no obstacles
