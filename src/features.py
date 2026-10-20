import scapy.all as scapy
import pandas as pd
import numpy as np
from collections import defaultdict

class NetworkFlowExtractor:
    def __init__(self):
        # Dictionary to hold flow data: {(src_ip, dst_ip, src_port, dst_port, proto): [packets]}
        self.flows = defaultdict(list)

    def _get_flow_key(self, packet):
        """Extracts the 5-tuple key, ensuring direction invariance for bi-directional flows."""
        if not (packet.haslayer(scapy.IP) and (packet.haslayer(scapy.TCP) or packet.haslayer(scapy.UDP))):
            return None
        
        proto = packet[scapy.IP].proto
        src_ip = packet[scapy.IP].src
        dst_ip = packet[scapy.IP].dst
        
        if packet.haslayer(scapy.TCP):
            src_port = packet[scapy.TCP].sport
            dst_port = packet[scapy.TCP].dport
        else:
            src_port = packet[scapy.UDP].sport
            dst_port = packet[scapy.UDP].dport

        # Order the tuple consistently so forward and backward packets belong to the same flow key
        if (src_ip, src_port) < (dst_ip, dst_port):
            return (src_ip, dst_ip, src_port, dst_port, proto, "fwd")
        else:
            return (dst_ip, src_ip, dst_port, src_port, proto, "bwd")

    def read_pcap(self, pcap_path):
        """Reads PCAP file and groups packets into bidirectional flows."""
        print(f"[*] Parsing {pcap_path}...")
        packets = scapy.sniff(offline=pcap_path)
        
        for pkt in packets:
            key_info = self._get_flow_key(pkt)
            if key_info is None:
                continue
            
            # Unpack key and append packet along with its inferred direction
            *flow_key, direction = key_info
            self.flows[tuple(flow_key)].append((pkt, direction))
        
        print(f"[+] Successfully extracted {len(self.flows)} unique network flows.")

    def extract_features(self):
        """Computes statistical and structural behavioral features for each flow."""
        flow_features_list = []

        for flow_key, packet_list in self.flows.items():
            src_ip, dst_ip, src_port, dst_port, proto = flow_key
            
            timestamps = [float(pkt.time) for pkt, _ in packet_list]
            pkt_lengths = [len(pkt) for pkt, _ in packet_list]
            
            fwd_lengths = [len(pkt) for pkt, dir in packet_list if dir == "fwd"]
            bwd_lengths = [len(pkt) for pkt, dir in packet_list if dir == "bwd"]

            # 1. Duration and Packet Timing (Inter-Arrival Times)
            duration = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0.0
            iats = np.diff(timestamps) if len(timestamps) > 1 else [0.0]

            # 2. Flag Counting (Focusing on TCP Flags for scanning/anomaly behaviors)
            syn_count = sum(1 for pkt, _ in packet_list if pkt.haslayer(scapy.TCP) and pkt[scapy.TCP].flags.S)
            ack_count = sum(1 for pkt, _ in packet_list if pkt.haslayer(scapy.TCP) and pkt[scapy.TCP].flags.A)
            fin_count = sum(1 for pkt, _ in packet_list if pkt.haslayer(scapy.TCP) and pkt[scapy.TCP].flags.F)
            rst_count = sum(1 for pkt, _ in packet_list if pkt.haslayer(scapy.TCP) and pkt[scapy.TCP].flags.R)
            psh_count = sum(1 for pkt, _ in packet_list if pkt.haslayer(scapy.TCP) and pkt[scapy.TCP].flags.P)

            # Compile feature dictionary
            features = {
                "src_port": src_port,
                "dst_port": dst_port,
                "protocol": proto,
                "flow_duration": duration,
                "total_fwd_pkts": len(fwd_lengths),
                "total_bwd_pkts": len(bwd_lengths),
                "total_fwd_len": sum(fwd_lengths),
                "total_bwd_len": sum(bwd_lengths),
                "fwd_pkt_len_max": max(fwd_lengths) if fwd_lengths else 0,
                "fwd_pkt_len_min": min(fwd_lengths) if fwd_lengths else 0,
                "fwd_pkt_len_mean": np.mean(fwd_lengths) if fwd_lengths else 0,
                "fwd_pkt_len_std": np.std(fwd_lengths) if fwd_lengths else 0,
                "bwd_pkt_len_max": max(bwd_lengths) if bwd_lengths else 0,
                "bwd_pkt_len_min": min(bwd_lengths) if bwd_lengths else 0,
                "bwd_pkt_len_mean": np.mean(bwd_lengths) if bwd_lengths else 0,
                "bwd_pkt_len_std": np.std(bwd_lengths) if bwd_lengths else 0,
                "flow_byts_s": sum(pkt_lengths) / duration if duration > 0 else 0,
                "flow_pkts_s": len(packet_list) / duration if duration > 0 else 0,
                "flow_iat_mean": np.mean(iats),
                "flow_iat_std": np.std(iats),
                "flow_iat_max": max(iats),
                "flow_iat_min": min(iats),
                "fwd_syn_cnt": syn_count,
                "fwd_ack_cnt": ack_count,
                "fwd_fin_cnt": fin_count,
                "fwd_rst_cnt": rst_count,
                "fwd_psh_cnt": psh_count,
                # Placeholders for mathematical scale padding to hit standard framework dimensions
                "param_pad_1": len(packet_list),
                "param_pad_2": np.median(pkt_lengths),
                "param_pad_3": np.percentile(pkt_lengths, 75),
                "param_pad_4": np.percentile(pkt_lengths, 25),
                "param_pad_5": np.var(pkt_lengths)
            }
            flow_features_list.append(features)

        return pd.DataFrame(flow_features_list)
