import csv
import os
import pandas as pd
from collections import defaultdict

from scapy.sessions import DefaultSession

from .features.context.packet_direction import PacketDirection
from .features.context.packet_flow_key import get_packet_flow_key
from .flow import Flow
from data_preprocess.pre_utils.payload_decoding import decode_payload

EXPIRED_UPDATE = 40
MACHINE_LEARNING_API = "http://localhost:8000/predict"
GARBAGE_COLLECT_PACKETS = 1000


class FlowSession(DefaultSession):
    """Creates a list of network flows."""

    def __init__(self, *args, **kwargs):
        self.flows = {}
        self.csv_line = 0
        self.df = None
        if self.output_mode == "flow":
            output = open(self.output_file, "w")
            self.csv_writer = csv.writer(output)
            self.df = None

        self.packets_count = 0

        self.clumped_flows_per_label = defaultdict(list)

        super(FlowSession, self).__init__(*args, **kwargs)

    def toPacketList(self):
        # Sniffer finished all the packets it needed to sniff.
        # It is not a good place for this, we need to somehow define a finish signal for AsyncSniffer
        self.garbage_collect(None)
        return super(FlowSession, self).toPacketList()

    def duplicate_flow(self, packet, direction):
        """Duplicates a flow.

        Args:
            flow (Flow): The flow to duplicate.

        Returns:
            Flow: The duplicated flow.

        """
        flow = Flow(packet, direction)
        flow.add_packet(packet, direction)
        data = flow.get_data()
        new_data = {}
        for key, value in data.items():
            if key in self.df.columns:
                new_data[key] = value
        if pd.concat([self.df, pd.DataFrame(new_data, index=[self.csv_line])]).duplicated().sum() != 0:
            return False
        return True

    def on_packet_received(self, packet):
        count = 0
        direction = PacketDirection.FORWARD

        if self.output_mode != "flow":
            if "TCP" not in packet:
                return
            elif "UDP" not in packet:
                return

        try:
            # Creates a key variable to check
            packet_flow_key = get_packet_flow_key(packet, direction)
            flow = self.flows.get((packet_flow_key, count))
        except Exception:
            return

        self.packets_count += 1

        # If there is no forward flow with a count of 0
        if flow is None:
            # There might be one of it in reverse
            direction = PacketDirection.REVERSE
            packet_flow_key = get_packet_flow_key(packet, direction)
            flow = self.flows.get((packet_flow_key, count))

        if flow is None:
            # If no flow exists create a new flow
            direction = PacketDirection.FORWARD
            flow = Flow(packet, direction)
            packet_flow_key = get_packet_flow_key(packet, direction)
            self.flows[(packet_flow_key, count)] = flow

        elif (packet.time - flow.latest_timestamp) > EXPIRED_UPDATE:
            # If the packet exists in the flow but the packet is sent
            # after too much of a delay than it is a part of a new flow.
            expired = EXPIRED_UPDATE
            while (packet.time - flow.latest_timestamp) > expired:
                count += 1
                expired += EXPIRED_UPDATE
                flow = self.flows.get((packet_flow_key, count))

                if flow is None:
                    flow = Flow(packet, direction)
                    self.flows[(packet_flow_key, count)] = flow
                    break
        elif "F" in str(packet.flags):
            # If it has FIN flag then early collect flow and continue
            flow.add_packet(packet, direction)
            self.garbage_collect(packet.time)
            return

        flow.add_packet(packet, direction)

        if self.packets_count % GARBAGE_COLLECT_PACKETS == 0 or (
            flow.duration > 120 and self.output_mode == "flow"
        ):
            self.garbage_collect(packet.time)

    def get_flows(self) -> list:
        return self.flows.values()

    def garbage_collect(self, latest_time) -> None:
        # TODO: Garbage Collection / Feature Extraction should have a separate thread
        # if not self.url_model:
        #     print("Garbage Collection Began. Flows = {}".format(len(self.flows)))
        keys = list(self.flows.keys())
        ports = [80, 8080, 443, 53, 20, 21, 25, 587, 465, 143, 110, 67, 68, 123, 5355, 139, 993, 445, 161, 162, 137, 138, 5353,
                  389, 636, 111, 135, 88, 30301, 30302, 30303, 30304, 30305, 5060, 6667, 1900, 43,
                 5500, 1883, 3702, 6881, 500, 5683, 873, 8995, -2, "XML", "TLS", "TLS/PKI", "LDAP", "PEncrypted", "XMRig_Mining", "Revenge-R", "BitTorrent-DHT",
                 "vnc", "Ethereum DevP2P", "IMAPS",
                 "Revenge-RAT", "InfoStealer", "BOT", "Suspicious-Custom-Botnet",
                 "Windows-CMD-Obfuscated", "VNC", "Windows-DLL-Exec",
                 "CUSTOM_POWERSHELL_C2", "custom_domain_tunnel"
                 "CustomC2-Base64", "custom_domain_tunnel", "TDS", "PRINTER"]
        for k in keys:
            flow = self.flows.get(k)
            if (
                latest_time is None
                or latest_time - flow.latest_timestamp > EXPIRED_UPDATE
                or flow.duration > 90
            ):
                flow_data = flow.get_data()
                dport = flow_data['dst_port']
                payload = flow_data['payload']
                if len(payload) < 50:
                    decoded = ""
                    port = dport
                else:
                    decoded, port = decode_payload(dport, payload)
                    try:
                        readable = decoded.split("Readable ASCII Data:")[-1]
                    except IndexError:
                        pass
                    except AttributeError:
                        print()
                flow_data['translation_port'] = port
                if len(decoded) < 200 and port not in ports:
                    decoded = ""
                else:
                    decoded = decoded.replace("\n", " ")
                flow_data['payload'] = decoded
                if self.csv_line == 0:
                    self.csv_writer.writerow(flow_data.keys())
                self.csv_writer.writerow(flow_data.values())
                self.csv_line += 1
                del self.flows[k]


def generate_session_class(output_mode, output_file, url_model):
    return type(
        "NewFlowSession",
        (FlowSession,),
        {
            "output_mode": output_mode,
            "output_file": output_file,
            "url_model": url_model,
        },
    )
