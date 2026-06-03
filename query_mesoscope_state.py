"""Manual MQTT probe that requests and prints a single Mesoscope state snapshot from the ScanImagePC.

Run this on the VRPC while runAcquisition is active and idle in its MQTT command loop on the ScanImagePC. It
publishes an empty-payload MesoscopeQueryState request, prints the MesoscopeState reply, and exits. Pass an
optional broker host as the first argument; it defaults to the local broker. This is a throwaway diagnostic
helper and is not part of the sollertia_experiment package.
"""

import sys
import json
import threading

import paho.mqtt.client as mqtt

_BROKER_HOST: str = "127.0.0.1"
"""The default MQTT broker host. The broker runs locally on the VRPC, so loopback is used unless overridden."""

_BROKER_PORT: int = 1883
"""The MQTT broker port shared by the Mesoscope control interface and the Unity Virtual Reality task."""

_QUERY_TOPIC: str = "MesoscopeQueryState"
"""The topic on which the VRPC requests a one-shot state snapshot from the ScanImagePC."""

_STATE_TOPIC: str = "MesoscopeState"
"""The topic on which the ScanImagePC publishes the requested state snapshot."""

_REPLY_TIMEOUT_S: float = 10.0
"""The maximum time, in seconds, to wait for the snapshot reply before giving up."""

_reply_received: threading.Event = threading.Event()
"""Signals the main thread that the snapshot reply arrived so it can stop the network loop."""


def _on_connect(client: mqtt.Client, userdata: object, flags: object, reason_code: object, properties: object) -> None:
    """Subscribes to the reply topic once the broker connection is established."""
    client.subscribe(topic=_STATE_TOPIC)


def _on_subscribe(client: mqtt.Client, userdata: object, mid: int, reason_codes: object, properties: object) -> None:
    """Publishes the empty-payload query only after the reply subscription is confirmed, avoiding a reply race."""
    client.publish(topic=_QUERY_TOPIC, payload="")


def _on_message(client: mqtt.Client, userdata: object, message: mqtt.MQTTMessage) -> None:
    """Prints the decoded state snapshot and signals the main thread that the reply arrived."""
    print(json.dumps(json.loads(message.payload), indent=2))
    _reply_received.set()


def main() -> int:
    """Connects to the broker, requests a single state snapshot, prints it, and returns a process exit code."""
    host = sys.argv[1] if len(sys.argv) > 1 else _BROKER_HOST

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = _on_connect
    client.on_subscribe = _on_subscribe
    client.on_message = _on_message
    client.connect(host=host, port=_BROKER_PORT, keepalive=60)

    client.loop_start()
    got_reply = _reply_received.wait(timeout=_REPLY_TIMEOUT_S)
    client.loop_stop()
    client.disconnect()

    if not got_reply:
        message = (
            f"No MesoscopeState reply within {_REPLY_TIMEOUT_S:.0f}s. Ensure runAcquisition is running and idle "
            f"in its command loop on the ScanImagePC, and that the broker host '{host}' is correct."
        )
        print(message, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
