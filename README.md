# Aurelius — Drone Streaming System

Real-time depth + segmentation pipeline over a drone video link.

```
[ Drone ]  →  sender.py  →  (network)  →  ground_station_inference.py  →  RTSP  →  viewer
```

---

## Drone (Sender)

### Start streaming — with raw frame saving
```bash
nohup python -u sender.py --save-dir ./flight_001 > flight_001.log 2>&1 &
echo $!
```

### Start streaming — no saving
```bash
nohup python -u sender.py > flight_001.log 2>&1 &
echo $!
```

> `nohup` keeps the process alive after SSH disconnects.  
> `-u` flushes output immediately so the log is readable in real time.  
> `echo $!` prints the PID — write it down to kill the process later.

### Watch the live log
```bash
tail -f flight_001.log
```

### Check the process is running
```bash
ps aux | grep sender
```

### Stop the stream
```bash
kill <PID>
```

---

## Ground Station (Receiver + Inference + RTSP)

### 1a — Start Docker MediaMTX (RTSP server)
Run this once before the ground station script. The RTSP stream will be
available as soon as this container is up, even before the drone connects.

```bash
docker run --rm -it --network=host bluenviron/mediamtx:latest
```

### 1b — Start Local MediaMTX (RTSP server)
Download and run MediaMTX locally without docker:

```bash
# Linux x86_64
wget https://github.com/bluenviron/mediamtx/releases/download/v1.18.1/mediamtx_v1.18.1_linux_amd64.tar.gz

# Extract archive
tar xzf mediamtx_*.tar.gz

# Start MediaMTX
./mediamtx
```

### 2 — Start the ground station
```bash
python ground_station_inference.py
```

With the original frame as a picture-in-picture overlay:
```bash
python ground_station_inference.py --show-original
```

Custom RTSP publish URL:
```bash
python ground_station_inference.py --rtsp-url rtsp://0.0.0.0:8554/drone
```

> The RTSP stream is live immediately at startup.  
> While no drone is connected the stream shows the **Aurelius** placeholder frame.  
> When the drone disconnects the receiver automatically re-listens — no restart needed.

---

## Viewer (Any Node on the Network)

Replace `SERVER_GROUND_ANY_INTERFACE` with the ground station IP (e.g. `192.168.1.10`).

### mpv
```bash
mpv --rtsp-transport=tcp rtsp://SERVER_GROUND_ANY_INTERFACE:8554/drone
```

### ffplay
```bash
ffplay rtsp://SERVER_GROUND_ANY_INTERFACE:8554/drone
```

### VLC
```bash
vlc rtsp://SERVER_GROUND_ANY_INTERFACE:8554/drone
```

---

## Startup Order

```
1. Ground station  →  docker run … mediamtx
2. Ground station  →  python ground_station_inference.py
3. Viewer          →  mpv rtsp://…
4. Drone           →  nohup python -u sender.py …
```

Steps 1–3 can be done before the drone is powered on — the placeholder frame
keeps the stream valid until the drone connects.

---

## Quick Reference

| Action | Command |
|---|---|
| Start sender (with save) | `nohup python -u sender.py --save-dir ./flight_001 > flight_001.log 2>&1 &` |
| Start sender (no save) | `nohup python -u sender.py > flight_001.log 2>&1 &` |
| Watch sender log | `tail -f flight_001.log` |
| Find sender PID | `ps aux \| grep sender` |
| Stop sender | `kill <PID>` |
| Start RTSP server | `docker run --rm -it --network=host bluenviron/mediamtx:latest` |
| Start ground station (no showing image)| `python ground_station_inference.py` |
| Start ground station (showing image) | `python ground_station_inference.py --show-original` |
| Watch stream | `mpv --rtsp-transport=tcp rtsp://<IP>:8554/drone` |