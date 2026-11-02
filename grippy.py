
import socket
import time

ROBOT_IP = "158.38.140.99"
ROBOT_PORT = 5891

THRESHOLD = 4  # Piksler
STEP = 1       # mm


def make_packet(data: str) -> bytes:
    length = len(data.encode("utf-8"))
    body = f"TMSVR,{length},{data},"

    checksum = 0
    for b in body.encode("utf-8"):
        checksum ^= b

    return f"${body}*{checksum:02X}\r\n".encode("utf-8")


def read_response(sock) -> str:
    response = bytearray()

    while not response.endswith(b"\r\n"):
        chunk = sock.recv(1)
        if not chunk:
            raise ConnectionError("Forbindelsen ble lukket")
        response.extend(chunk)

    return response.decode("utf-8", errors="replace").strip()


def send_command(sock, payload: str) -> str:
    packet = make_packet(payload)
    print("Sender:", packet.decode().strip())

    sock.sendall(packet)
    response = read_response(sock)

    print("Svar fra robot:", response)
    return response


def get_camera_error(sock):
    # Les X-avvik
   # Les X-avvik fra kamera
    response_x = send_command(sock, "T2,12,g_diffX")
    diffX = float(response_x)

# Les Y-avvik fra kamera
    response_y = send_command(sock, "T2,12,g_diffY")
    diffY = float(response_y)

   
    return response_x, response_y


def calculate_movement(diff):
    if diff > THRESHOLD:
        return -STEP
    elif diff < -THRESHOLD:
        return STEP
    else:
        return 0


try:
    with socket.create_connection(
        (ROBOT_IP, ROBOT_PORT), timeout=3
    ) as sock:

        while True:

            # 1. Les kameraverdier
            response_x, response_y = get_camera_error(sock)

            print("\nX:", response_x)
            print("Y:", response_y)

            # 2. Behandle verdiene
            # Her må råsvarene parses til tall før beregning.
            # Eksempel:
            diffX = float(response_x)
            diffY = float(response_y)

            # 3. Beregn bevegelse
            moveX = calculate_movement(diffX)
            moveY = calculate_movement(diffY)

            print("Bevegelse X:", moveX, "mm")
            print("Bevegelse Y:", moveY, "mm")

            # 4. Skriv bevegelsesverdier til roboten
            if moveX != 0:
                send_command(
                    sock,
                    f"T1,2,diffX={moveX}"
                )

            if moveY != 0:
                send_command(
                    sock,
                    f"T1,2,diffY={moveY}"
                )

            time.sleep(0.1)


except (OSError, ConnectionError, ValueError) as error:
    print("Kommunikasjonsfeil:", error)
