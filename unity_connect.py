import socket
import time

#communication
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 5052)

while True:
    sock.sendto(str.encode("data_to_Unity"), serverAddressPort)
    time.sleep(0.1)