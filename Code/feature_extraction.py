# Standard library imports
from collections import defaultdict
from datetime import datetime
import os
import json
import traceback
import warnings
import concurrent.futures
from tqdm import tqdm  # For progress bars
import sys

# Third-party imports
import pandas as pd
import numpy as np
import ipaddress
from scipy.stats import entropy, skew
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from scapy.all import rdpcap, PcapReader, IP, TCP, UDP, raw
import matplotlib.pyplot as plt  # For feature importance visualization

# GUI imports
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Constants
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FEATURE_FILE = os.path.join(os.path.dirname(__file__), "feature_dict.json")

# Global configuration
CONFIG = {
    "BATCH_SIZE": int(os.environ.get('BATCH_SIZE', '100000')),
    "DEBUG_MODE": os.environ.get('DEBUG_MODE', '0') == '1',
    "KEEP_TEMP": os.environ.get('KEEP_TEMP_FILES', '0') == '1',
    "ANALYZE_FEATURES": os.environ.get('ANALYZE_FEATURES', '1') == '1',
    "NUM_THREADS": int(os.environ.get('NUM_THREADS', '4'))
}

# Suppress warnings
if not CONFIG["DEBUG_MODE"]:
    warnings.filterwarnings('ignore')

def manage_features(df):
    """Manages features based on the feature dictionary and handles missing features"""
    # Get current features excluding 'attack'
    current_features = [col for col in df.columns if col != "attack"]
    
    # Load or create feature set
    if os.path.exists(FEATURE_FILE):
        with open(FEATURE_FILE, "r") as f:
            stored_features = json.load(f)
            
        # Add any new features to the stored set
        new_features = list(set(current_features) - set(stored_features))
        if new_features:
            stored_features.extend(new_features)
            with open(FEATURE_FILE, "w") as f:
                json.dump(stored_features, f, indent=4)
            print(f"✅ Feature set updated with {len(new_features)} new features")
            
        # Add missing features to dataframe
        for feature in stored_features:
            if feature not in df.columns and feature != "attack":
                df[feature] = 0
                
        # Reorder columns to match stored feature set
        available_features = [f for f in stored_features if f in df.columns]
        attack_cols = ["attack"] if "attack" in df.columns else []
        df = df[available_features + attack_cols]
    else:
        # First time running - save current features
        with open(FEATURE_FILE, "w") as f:
            json.dump(current_features, f, indent=4)
        print(f"✅ Initial feature set created with {len(current_features)} features")
        
    return df

def extract_pcap_features(pcap_file, output_csv, attack_type="any"):
    """
    Extract network features from pcap files with support for all attack types
    
    Parameters:
    - pcap_file: Path to the pcap file
    - output_csv: Path to save the extracted features
    - attack_type: Type of attack to optimize extraction for (ddos, memcrashedspoofer, portscan, or any)
    
    Returns:
    - Boolean indicating success or failure
    """
    try:
        print(f"📂 Reading pcap file: {pcap_file}")
        print(f"📊 Extracting features optimized for: {attack_type.upper() if attack_type else 'ALL'} attacks")
        
        # Check file size before loading for memory estimate
        file_size_mb = os.path.getsize(pcap_file) / (1024 * 1024)
        print(f"📊 PCAP file size: {file_size_mb:.2f} MB")
        
        # Adjust batch size based on file size and attack type
        batch_size = CONFIG["BATCH_SIZE"]
        if (file_size_mb > 500 and attack_type == "portscan"):  # PortScan needs smaller batches for large files
            batch_size = min(batch_size, 50000)
            print(f"🔧 Large file detected for PortScan. Adjusting batch size to {batch_size}")
            
        # Load packets with optimized approach
        print("⏳ Loading packets... This may take some time for large files...")
        
        # Count total packets for progress tracking if possible
        try:
            with PcapReader(pcap_file) as pcap_reader:
                total_packets = sum(1 for _ in pcap_reader)
            print(f"📦 Total packets: {total_packets:,}")
        except Exception as e:
            print(f"⚠️ Unable to count packets: {e}")
            print("⚠️ Will process without packet count")
            total_packets = None
            
        # Now load for actual processing
        packets = rdpcap(pcap_file)
        packet_count = len(packets)
        print(f"✅ Successfully loaded {packet_count:,} packets")
        
        # Initialize data structures for all attack types
        data = []
        
        # Common data structures for needed features
        flow_stats = {}
        flag_stats = {}
        connection_stats = {}
        
        # PortScan-specific structures
        sliding_windows = {}  # For 5-second sliding windows
        port_sequence = {}    # For tracking sequential port scans
        
        # Batch process packets for memory efficiency
        batch_size = min(batch_size, packet_count)
        batch_count = (packet_count + batch_size - 1) // batch_size
        
        print(f"🔄 Processing {batch_count:,} batches of up to {batch_size:,} packets each...")
        
        # Function to process a single batch with support for needed features
        def process_batch(batch_idx):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, packet_count)
            batch_data = []
            
            # Common structures for needed features
            batch_flow_stats = {}
            batch_flag_stats = {}
            batch_connections = {}
            
            # PortScan-specific structures
            batch_sliding_windows = {}
            batch_port_sequence = {}
            
            # Process packets in this batch
            for pkt in packets[batch_start:batch_end]:
                packet_info = {}
                
                # Timestamp processing (common)
                packet_time = pkt.time if hasattr(pkt, "time") else 0
                packet_info["frame.time_epoch"] = packet_time
                second = int(packet_time)
                
                # Initialize dictionaries for this second (common to all attack types)
                if second not in batch_flow_stats:
                    batch_flow_stats[second] = {
                        "packet_count": 0, 
                        "byte_count": 0, 
                        "src_ips": set(), 
                        "dst_ips": set(),
                        "timestamps": [],
                        "unique_dst_ports": set(),
                        "memcached_packet_count": 0,
                    }
                    batch_flag_stats[second] = {
                        "syn": 0, "total": 0
                    }
                    batch_connections[second] = {
                        "connections": set(), 
                    }
                
                # Process IP Layer (common)
                src_ip = dst_ip = "0.0.0.0"
                proto = 0
                
                if pkt.haslayer(IP):
                    src_ip = pkt[IP].src
                    dst_ip = pkt[IP].dst
                    proto = pkt[IP].proto
                
                packet_info.update({
                    "ip.src": src_ip,
                    "ip.dst": dst_ip,
                    "ip.proto": proto,
                })
                
                # Process TCP/UDP Layer (common with attack-specific logic)
                src_port = dst_port = 0
                tcp_flags = 0
                packet_info["tcp.flags.syn"] = 0
                
                if pkt.haslayer(TCP):
                    src_port = int(pkt[TCP].sport)
                    dst_port = int(pkt[TCP].dport)
                    tcp_flags = int(pkt[TCP].flags)
                    
                    # Process TCP flags
                    is_syn = bool(tcp_flags & 0x02)
                    
                    packet_info["tcp.flags.syn"] = 1 if is_syn else 0
                    
                    # Update flag statistics
                    batch_flag_stats[second]["total"] += 1
                    if is_syn:
                        batch_flag_stats[second]["syn"] += 1
                        
                elif pkt.haslayer(UDP):
                    src_port = int(pkt[UDP].sport)
                    dst_port = int(pkt[UDP].dport)
                    
                    # Add header to payload ratio calculation
                    try:
                        header_size = len(pkt) - len(pkt[UDP].payload)
                        payload_size = len(pkt[UDP].payload)
                        if payload_size > 0:
                            header_to_payload = header_size / payload_size
                            packet_info["header_payload_ratio"] = header_to_payload
                        else:
                            packet_info["header_payload_ratio"] = 1.0
                    except:
                        packet_info["header_payload_ratio"] = 1.0
                    
                    # Update memcached packet count if relevant
                    if src_port == 11211 or dst_port == 11211:
                        batch_flow_stats[second]["memcached_packet_count"] += 1
                
                # Add port information
                packet_info["srcport"] = src_port
                packet_info["dstport"] = dst_port
                packet_info["flags"] = tcp_flags
                
                # Process packet length
                pkt_len = len(pkt)
                packet_info["packet.length"] = pkt_len
                packet_info["frame.len"] = pkt_len  # Also store as frame.len
                
                # Update flow statistics
                batch_flow_stats[second]["packet_count"] += 1
                batch_flow_stats[second]["byte_count"] += pkt_len
                batch_flow_stats[second]["src_ips"].add(src_ip)
                batch_flow_stats[second]["dst_ips"].add(dst_ip)
                batch_flow_stats[second]["timestamps"].append(packet_time)
                batch_flow_stats[second]["unique_dst_ports"].add(dst_port)
                
                # Track connection tuples
                connection_tuple = (src_ip, dst_ip, src_port, dst_port)
                batch_connections[second]["connections"].add(connection_tuple)
                
                # PortScan-specific processing
                if attack_type in ["portscan", "any"]:
                    # Track port sequence for each src-dst IP pair
                    ip_pair = f"{src_ip}-{dst_ip}"
                    if ip_pair not in batch_port_sequence:
                        batch_port_sequence[ip_pair] = {
                            "last_port": 0, 
                            "sequential_count": 0, 
                            "ports": set()
                        }
                    
                    # Count unique ports and detect sequential port scans
                    current_port = dst_port
                    batch_port_sequence[ip_pair]["ports"].add(current_port)
                    
                    # Simple sequential port detection
                    if abs(current_port - batch_port_sequence[ip_pair]["last_port"]) == 1:
                        batch_port_sequence[ip_pair]["sequential_count"] += 1
                    
                    batch_port_sequence[ip_pair]["last_port"] = current_port
                    
                    # 5-second sliding window tracking
                    window_key = int(packet_time / 5) * 5  # Round to 5-second window
                    if window_key not in batch_sliding_windows:
                        batch_sliding_windows[window_key] = {
                            "packet_count": 0,
                            "unique_src_ips": set(),
                            "unique_dst_ips": set(),
                            "unique_dst_ports": set(),
                        }
                    batch_sliding_windows[window_key]["packet_count"] += 1
                    batch_sliding_windows[window_key]["unique_src_ips"].add(src_ip)
                    batch_sliding_windows[window_key]["unique_dst_ips"].add(dst_ip)
                    batch_sliding_windows[window_key]["unique_dst_ports"].add(dst_port)
                    
                    # Add PortScan-specific features to packet info
                    packet_info["src_dst_port_diversity"] = len(batch_port_sequence[ip_pair]["ports"])
                    
                    # Calculate port scan density if we have ports
                    if len(batch_port_sequence[ip_pair]["ports"]) > 0:
                        min_port = min(batch_port_sequence[ip_pair]["ports"]) 
                        max_port = max(batch_port_sequence[ip_pair]["ports"])
                        port_range = max(1, max_port - min_port + 1)
                        port_density = len(batch_port_sequence[ip_pair]["ports"]) / port_range
                        packet_info["port_scan_density"] = port_density
                    else:
                        packet_info["port_scan_density"] = 0
                
                # Initialize injected_packets feature to 0 (to be updated later if needed)
                packet_info["injected_packets"] = 0
                
                # Add the processed packet to results
                batch_data.append(packet_info)
            
            return {
                'data': batch_data,
                'flow_stats': batch_flow_stats,
                'flag_stats': batch_flag_stats,
                'connections': batch_connections,
                'sliding_windows': batch_sliding_windows,
                'port_sequence': batch_port_sequence
            }

        # Use parallel processing if more than one batch and not in debug mode
        if batch_count > 1 and not CONFIG["DEBUG_MODE"]:
            print(f"🔄 Using parallel processing with {min(CONFIG['NUM_THREADS'], batch_count)} threads")
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(CONFIG['NUM_THREADS'], batch_count)) as executor:
                batch_results = list(tqdm(
                    executor.map(process_batch, range(batch_count)),
                    total=batch_count,
                    desc="Processing batches"
                ))
                
            # Merge batch results efficiently
            print("🔄 Merging batch results...")
            for batch_result in tqdm(batch_results, desc="Merging results"):
                data.extend(batch_result['data'])
                
                # Merge common dictionaries
                for second, stats in batch_result['flow_stats'].items():
                    if second not in flow_stats:
                        flow_stats[second] = {
                            "packet_count": 0, "byte_count": 0, "src_ips": set(), 
                            "dst_ips": set(), "timestamps": [], "unique_dst_ports": set(),
                            "memcached_packet_count": 0
                        }
                    flow_stats[second]["packet_count"] += stats["packet_count"]
                    flow_stats[second]["byte_count"] += stats["byte_count"]
                    flow_stats[second]["src_ips"].update(stats["src_ips"])
                    flow_stats[second]["dst_ips"].update(stats["dst_ips"])
                    flow_stats[second]["timestamps"].extend(stats["timestamps"])
                    flow_stats[second]["unique_dst_ports"].update(stats["unique_dst_ports"])
                    
                    # Memcached-specific
                    if "memcached_packet_count" in stats:
                        flow_stats[second]["memcached_packet_count"] = flow_stats[second].get("memcached_packet_count", 0) + stats["memcached_packet_count"]
                
                for second, stats in batch_result['flag_stats'].items():
                    if second not in flag_stats:
                        flag_stats[second] = {"syn": 0, "total": 0}
                    flag_stats[second]["syn"] += stats["syn"]
                    flag_stats[second]["total"] += stats["total"]
                
                for second, stats in batch_result['connections'].items():
                    if second not in connection_stats:
                        connection_stats[second] = {"connections": set()}
                    connection_stats[second]["connections"].update(stats["connections"])
                
                # Merge PortScan-specific structures
                if attack_type in ["portscan", "any"]:
                    # Merge sliding windows
                    for window_key, window_stats in batch_result['sliding_windows'].items():
                        if window_key not in sliding_windows:
                            sliding_windows[window_key] = {
                                "packet_count": 0,
                                "unique_src_ips": set(),
                                "unique_dst_ips": set(),
                                "unique_dst_ports": set(),
                            }
                        sliding_windows[window_key]["packet_count"] += window_stats["packet_count"]
                        sliding_windows[window_key]["unique_src_ips"].update(window_stats["unique_src_ips"])
                        sliding_windows[window_key]["unique_dst_ips"].update(window_stats["unique_dst_ips"])
                        sliding_windows[window_key]["unique_dst_ports"].update(window_stats["unique_dst_ports"])
                    
                    # Merge port sequences
                    for ip_pair, seq_stats in batch_result['port_sequence'].items():
                        if ip_pair not in port_sequence:
                            port_sequence[ip_pair] = {
                                "last_port": seq_stats["last_port"],
                                "sequential_count": seq_stats["sequential_count"],
                                "ports": seq_stats["ports"].copy()
                            }
                        else:
                            port_sequence[ip_pair]["last_port"] = seq_stats["last_port"]
                            port_sequence[ip_pair]["sequential_count"] += seq_stats["sequential_count"]
                            port_sequence[ip_pair]["ports"].update(seq_stats["ports"])
        else:
            print("🔄 Using sequential processing...")
            # Process each batch sequentially with a progress bar
            for batch_idx in tqdm(range(batch_count), desc="Processing batches"):
                batch_result = process_batch(batch_idx)
                
                data.extend(batch_result['data'])
                
                # Use the same merging code as in the parallel section
                # Merge common dictionaries
                for second, stats in batch_result['flow_stats'].items():
                    if second not in flow_stats:
                        flow_stats[second] = {
                            "packet_count": 0, "byte_count": 0, "src_ips": set(), 
                            "dst_ips": set(), "timestamps": [], "unique_dst_ports": set(),
                            "memcached_packet_count": 0
                        }
                    flow_stats[second]["packet_count"] += stats["packet_count"]
                    flow_stats[second]["byte_count"] += stats["byte_count"]
                    flow_stats[second]["src_ips"].update(stats["src_ips"])
                    flow_stats[second]["dst_ips"].update(stats["dst_ips"])
                    flow_stats[second]["timestamps"].extend(stats["timestamps"])
                    flow_stats[second]["unique_dst_ports"].update(stats["unique_dst_ports"])
                    
                    # Memcached-specific
                    if "memcached_packet_count" in stats:
                        flow_stats[second]["memcached_packet_count"] = flow_stats[second].get("memcached_packet_count", 0) + stats["memcached_packet_count"]
                
                for second, stats in batch_result['flag_stats'].items():
                    if second not in flag_stats:
                        flag_stats[second] = {"syn": 0, "total": 0}
                    flag_stats[second]["syn"] += stats["syn"]
                    flag_stats[second]["total"] += stats["total"]
                
                for second, stats in batch_result['connections'].items():
                    if second not in connection_stats:
                        connection_stats[second] = {"connections": set()}
                    connection_stats[second]["connections"].update(stats["connections"])
                
                # Merge PortScan-specific structures
                if attack_type in ["portscan", "any"]:
                    # Merge sliding windows
                    for window_key, window_stats in batch_result['sliding_windows'].items():
                        if window_key not in sliding_windows:
                            sliding_windows[window_key] = {
                                "packet_count": 0,
                                "unique_src_ips": set(),
                                "unique_dst_ips": set(),
                                "unique_dst_ports": set(),
                            }
                        sliding_windows[window_key]["packet_count"] += window_stats["packet_count"]
                        sliding_windows[window_key]["unique_src_ips"].update(window_stats["unique_src_ips"])
                        sliding_windows[window_key]["unique_dst_ips"].update(window_stats["unique_dst_ips"])
                        sliding_windows[window_key]["unique_dst_ports"].update(window_stats["unique_dst_ports"])
                    
                    # Merge port sequences
                    for ip_pair, seq_stats in batch_result['port_sequence'].items():
                        if ip_pair not in port_sequence:
                            port_sequence[ip_pair] = {
                                "last_port": seq_stats["last_port"],
                                "sequential_count": seq_stats["sequential_count"],
                                "ports": seq_stats["ports"].copy()
                            }
                        else:
                            port_sequence[ip_pair]["last_port"] = seq_stats["last_port"]
                            port_sequence[ip_pair]["sequential_count"] += seq_stats["sequential_count"]
                            port_sequence[ip_pair]["ports"].update(seq_stats["ports"])

        print("🔄 Post-processing collected data...")
        
        # DataFrame creation and processing
        print("🔄 Creating DataFrame...")
        df = pd.DataFrame(data)
        
        if len(df) == 0:
            print("⚠️ No packets processed. Check the PCAP file.")
            return False
        
        # Pre-calculate values needed multiple times
        print("🔄 Calculating derived features...")
        
        # Only create temporary columns once
        df["frame.time_epoch_temp"] = pd.to_numeric(df["frame.time_epoch"], errors="coerce")
        frame_time_int = df["frame.time_epoch_temp"].astype(int)
        
        # Add common flow-based features
        print("🔄 Adding flow-based features...")
        df["packet.rate"] = frame_time_int.map(lambda x: flow_stats.get(x, {"packet_count": 0})["packet_count"])
        df["byte.count"] = frame_time_int.map(lambda x: flow_stats.get(x, {"byte_count": 0})["byte_count"])
        
        # Add memcached packet rate
        df["memcached.packet.rate"] = frame_time_int.map(
            lambda x: flow_stats.get(x, {"memcached_packet_count": 0})["memcached_packet_count"]
        )
        
        # Add connection-based features
        print("🔄 Adding connection-based features...")
        df["connection_count"] = frame_time_int.map(lambda x: len(connection_stats.get(x, {"connections": set()})["connections"]))
        
        # Add flag-based features
        print("🔄 Adding flag-based features...")
        df["syn_ratio"] = frame_time_int.map(lambda x: get_flag_ratio(x, flag_stats, "syn"))
        
        # Add port entropy features
        print("🔄 Adding port entropy features...")
        src_port_entropy, dst_port_entropy = optimize_port_entropy_calculations(frame_time_int, data)
        df["src_port_entropy"] = frame_time_int.map(lambda x: src_port_entropy.get(x, 0.0))
        df["dst_port_entropy"] = frame_time_int.map(lambda x: dst_port_entropy.get(x, 0.0))
        
        # Add PortScan-specific features
        if attack_type in ["portscan", "any"]:
            print("🔄 Adding window-based features...")
            df["window.key"] = df["frame.time_epoch"].apply(lambda x: int(x / 5) * 5)
            df["window.unique.dst.ports"] = df["window.key"].map(
                lambda x: len(sliding_windows.get(x, {"unique_dst_ports": set()})["unique_dst_ports"]))
            
            # Calculate port-to-packet ratio (high ratio indicates port scanning)
            df["ports_per_packet_ratio"] = df["window.unique.dst.ports"] / df["packet.rate"].apply(lambda x: max(x, 1))
        
        # Clean up temporary columns
        df.drop(columns=["frame.time_epoch_temp"], errors='ignore', inplace=True)
        if "window.key" in df.columns:
            df.drop(columns=["window.key"], errors='ignore', inplace=True)
        
        # Convert to numeric types
        numeric_cols = df.columns.difference(["ip.src", "ip.dst"])
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        
        # Fill NaN values
        df.fillna(0, inplace=True)
        
        # Save the result
        print(f"🔄 Saving extracted features to {os.path.basename(output_csv)}...")
        
        # Create output directory if needed
        output_dir = os.path.dirname(output_csv)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        df.to_csv(output_csv, index=False)
        
        # Print summary statistics
        print(f"✅ Extracted features saved to {output_csv}")
        print(f"📊 Total packets processed: {len(df):,}")
        print(f"📊 Total features extracted: {len(df.columns):,}")
        
        # Report memory usage
        memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)
        print(f"📊 DataFrame memory usage: {memory_usage:.2f} MB")
        
        return True
            
    except Exception as e:
        import traceback
        print(f"❌ Error in extract_pcap_features: {e}")
        traceback.print_exc()
        return False

def get_flag_ratio(second, flag_stats, flag_type):
    """Calculate the ratio of a specific flag type to total flags"""
    if second in flag_stats and flag_stats[second]["total"] > 0:
        return flag_stats[second][flag_type] / flag_stats[second]["total"]
    return 0

def detect_memcached_server(df, threshold=10):
    """
    Detect potential memcached servers in traffic based on port 11211 usage
    Returns a set of server IPs
    """
    memcached_servers = set()
    
    # First identify all IPs using the memcached port (11211)
    if "srcport" in df.columns and "dstport" in df.columns:
        # Extract all IPs sending from port 11211
        src_memcached = df[df["srcport"] == 11211]["ip.src"].unique()
        for ip in src_memcached:
            memcached_servers.add(ip)
            
        # Extract all IPs receiving on port 11211
        dst_memcached = df[df["dstport"] == 11211]["ip.dst"].unique()
        for ip in dst_memcached:
            memcached_servers.add(ip)
    
    return memcached_servers

def detect_port_scan_sequences(df, threshold=5):
    """
    Detect port scanning patterns in the dataset
    Returns a dictionary of src-dst IP pairs with sequential port counts
    """
    scanning_pairs = {}
    port_sequences = {}
    
    # Extract unique source-destination IP pairs
    ip_pairs = df[["ip.src", "ip.dst"]].drop_duplicates().values
    
    for src_ip, dst_ip in ip_pairs:
        # Get all packets between these IPs, sorted by time
        pair_df = df[(df["ip.src"] == src_ip) & (df["ip.dst"] == dst_ip)].sort_values("frame.time_epoch")
        
        # Skip if not enough packets
        if len(pair_df) < threshold:
            continue
            
        # Check for sequential port usage
        dst_ports = pair_df["dstport"].values
        sequential_count = 0
        
        for i in range(1, len(dst_ports)):
            if abs(dst_ports[i] - dst_ports[i-1]) == 1:
                sequential_count += 1
        
        # If we have sequential ports above threshold, record it
        if sequential_count >= threshold:
            key = f"{src_ip}-{dst_ip}"
            scanning_pairs[key] = sequential_count
            port_sequences[key] = sorted(pair_df["dstport"].unique())
    
    return scanning_pairs, port_sequences

def add_sliding_window_features(df, window_size=5):
    """Add sliding window features for port scan detection"""
    print("🔄 Adding sliding window features for PortScan detection...")
    
    # Create window key (5-second intervals by default)
    df["window_key"] = df["frame.time_epoch"].apply(lambda x: int(x / window_size) * window_size)
    
    # Initialize window features
    sliding_windows = {}
    
    # Calculate window statistics
    for window_key, group in df.groupby("window_key"):
        sliding_windows[window_key] = {
            "packet_count": len(group),
            "unique_src_ips": len(group["ip.src"].unique()),
            "unique_dst_ips": len(group["ip.dst"].unique()), 
            "unique_dst_ports": len(group["dstport"].unique()),
            "syn_count": group["tcp.flags.syn"].sum()
        }
    
    # Map these statistics back to the dataframe
    df["window.packet.rate"] = df["window_key"].map(lambda x: sliding_windows.get(x, {}).get("packet_count", 0))
    df["window.unique.src.ips"] = df["window_key"].map(lambda x: sliding_windows.get(x, {}).get("unique_src_ips", 0))
    df["window.unique.dst.ips"] = df["window_key"].map(lambda x: sliding_windows.get(x, {}).get("unique_dst_ips", 0))
    df["window.unique.dst.ports"] = df["window_key"].map(lambda x: sliding_windows.get(x, {}).get("unique_dst_ports", 0))
    df["window.syn.rate"] = df["window_key"].map(lambda x: sliding_windows.get(x, {}).get("syn_count", 0))
    
    # Clean up temporary column
    df.drop(columns=["window_key"], inplace=True)
    
    return df

def process_pcap_to_final(pcap_file, xml_file, final_csv, timestamp, output_directory, attack_type="any"):
    """
    Process PCAP and XML files into final dataset with progress tracking and memory optimization.
    """
    try:
        start_time = datetime.now()
        
        # Create organized directory structure
        temp_dir = os.path.join(output_directory, "temporary_files", timestamp)
        os.makedirs(temp_dir, exist_ok=True)

        # Define intermediate files with attack-specific naming
        extracted_csv = os.path.join(temp_dir, f"extracted_{attack_type}_data_{timestamp}.csv")
        labeled_csv = os.path.join(temp_dir, f"labeled_{attack_type}_data_{timestamp}.csv")
        numerical_csv = os.path.join(temp_dir, f"numerical_{attack_type}_data_{timestamp}.csv")
        normalized_csv = os.path.join(temp_dir, f"normalized_{attack_type}_data_{timestamp}.csv")

        print("\n" + "="*80)
        print(f"🚀 Starting full data processing pipeline for {attack_type.upper()} Attack...")
        print(f"📂 Input PCAP: {os.path.basename(pcap_file)}")
        print(f"📂 Input XML: {os.path.basename(xml_file)}")
        print(f"📂 Output CSV: {os.path.basename(final_csv)}")
        print("="*80 + "\n")

        # Processing pipeline with status checks
        pipeline_steps = [
            ("Feature extraction", lambda: extract_pcap_features(pcap_file, extracted_csv, attack_type)),
            ("Attack labeling", lambda: import_and_run_labeling(extracted_csv, xml_file, labeled_csv, attack_type)),
            ("Numerical conversion", lambda: import_and_run_numerical_conversion(labeled_csv, numerical_csv)),
            ("Data normalization", lambda: import_and_run_normalize_data(numerical_csv, normalized_csv))
        ]
        
        # Execute pipeline with progress tracking
        for step_name, step_func in pipeline_steps:
            step_start = datetime.now()
            print(f"\n🔄 {step_name} started at {step_start.strftime('%H:%M:%S')}...")
            
            # Execute step and check result
            success = step_func()
            step_end = datetime.now()
            step_duration = (step_end - step_start).total_seconds()
            
            if success is False:  # Explicitly check for False (not just falsy values)
                print(f"❌ {step_name} failed after {step_duration:.2f} seconds")
                return False
                
            print(f"✅ {step_name} completed in {step_duration:.2f} seconds")
            
            # Memory optimization - clear cache after heavy operations
            import gc
            gc.collect()
        
        print("\n🔄 Finalizing dataset...")
        
        try:
            # Load the normalized data
            df = pd.read_csv(normalized_csv)
            
            # Manage features
            df = manage_features(df)
            print("✅ Feature management completed")
            
            # Final data validation and cleaning
            print("🔍 Performing final data validation...")
            
            # Check for unusually large or NaN values that might indicate issues
            numeric_columns = df.select_dtypes(include=['number']).columns
            nan_counts = df[numeric_columns].isna().sum().sum()
            if nan_counts > 0:
                print(f"⚠️ Found {nan_counts} NaN values in the dataset. Filling with zeros.")
            
            # Handle missing values and save final result
            df.fillna(0, inplace=True)
            
            # Verify attack labels are correct for the specific attack type
            if attack_type == "memcrashedspoofer" and "attack" in df.columns:
                # Ensure Memcached attacks are labeled correctly (value 2)
                attack_count = df["attack"].sum()
                if attack_count > 0:
                    # Check if we need to update labels to standardized values
                    if len(df[df["attack"] > 0]["attack"].unique()) == 1 and df[df["attack"] > 0]["attack"].unique()[0] == 1:
                        print("⚠️ Updating Memcrashed attack labels from 1 to 2")
                        df.loc[df["attack"] == 1, "attack"] = 2
            
            if attack_type == "portscan" and "attack" in df.columns:
                # Ensure PortScan attacks are labeled correctly (value 3)
                attack_count = df["attack"].sum()
                if attack_count > 0:
                    # Check if we need to update labels to standardized values
                    if len(df[df["attack"] > 0]["attack"].unique()) == 1 and df[df["attack"] > 0]["attack"].unique()[0] == 1:
                        print("⚠️ Updating PortScan attack labels from 1 to 3")
                        df.loc[df["attack"] == 1, "attack"] = 3
            
            # Verify all feature columns are present and correct types
            print(f"📊 Final dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
            print(f"📊 Attack distribution: {df['attack'].value_counts().to_dict()}")
            
            # Save final dataset
            output_dir = os.path.dirname(final_csv)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                
            df.to_csv(final_csv, index=False)
            
            end_time = datetime.now()
            total_time = (end_time - start_time).total_seconds()
            
            print(f"\n🎯 Final processed dataset saved as {final_csv}")
            print(f"⏱️ Total processing time: {total_time:.2f} seconds")
            
            # Return the path to the final CSV for potential feature importance analysis
            return final_csv
            
        except Exception as e:
            print(f"❌ Error in final processing: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"❌ Error in process_pcap_to_final: {e}")
        import traceback
        traceback.print_exc()
        return False

def import_and_run_labeling(csv_file, xml_file, output_csv, attack_type="any"):
    """Import the labeling functionality from feature_labelling and run it."""
    try:
        # Try to import the labeling module
        import feature_labelling
        return feature_labelling.label_attack_data(csv_file, xml_file, output_csv, attack_type)
    except ImportError:
        print("❌ Could not import feature_labelling module. Make sure it's in the same directory.")
        return False
    except Exception as e:
        print(f"❌ Error running labeling: {str(e)}")
        traceback.print_exc()
        return False

def import_and_run_numerical_conversion(labeled_csv, numerical_csv):
    """Import the numerical conversion functionality from feature_labelling and run it."""
    try:
        # Try to import the numerical conversion module
        import feature_labelling
        return feature_labelling.convert_to_numerical(labeled_csv, numerical_csv)
    except ImportError:
        print("❌ Could not import feature_labelling module. Make sure it's in the same directory.")
        return False
    except Exception as e:
        print(f"❌ Error running numerical conversion: {str(e)}")
        traceback.print_exc()
        return False

def import_and_run_normalize_data(numerical_csv, normalized_csv):
    """Import the normalization functionality from feature_labelling and run it."""
    try:
        # Try to import the labeling module
        import feature_labelling
        return feature_labelling.normalize_data(numerical_csv, normalized_csv)
    except ImportError:
        print("❌ Could not import feature_labelling module. Make sure it's in the same directory.")
        return False
    except Exception as e:
        print(f"❌ Error running normalization: {str(e)}")
        traceback.print_exc()
        return False

def optimize_port_entropy_calculations(frame_time_int, data):
    """Optimized version of port entropy calculations with progress bar"""
    print("🔄 Adding port entropy features (optimized)...")
    
    # Pre-compute all values at once for each second
    src_port_entropy = {}
    dst_port_entropy = {}
    
    # Group data by second for faster processing
    seconds_data = {}
    for packet_info in data:
        second = int(packet_info["frame.time_epoch"])
        if second not in seconds_data:
            seconds_data[second] = {"src_ports": [], "dst_ports": []}
        seconds_data[second]["src_ports"].append(packet_info["srcport"])
        seconds_data[second]["dst_ports"].append(packet_info["dstport"])
    
    # Get unique seconds for progress tracking
    unique_seconds = sorted(seconds_data.keys())
    
    # Process all seconds in one loop with progress bar
    for second in tqdm(unique_seconds, desc="Calculating port entropy statistics"):
        if second in seconds_data:
            # Source port entropy
            src_ports = seconds_data[second]["src_ports"]
            if src_ports:
                values, counts = np.unique(src_ports, return_counts=True)
                src_port_entropy[second] = float(entropy(counts))
            else:
                src_port_entropy[second] = 0.0
                
            # Destination port entropy
            dst_ports = seconds_data[second]["dst_ports"]
            if dst_ports:
                values, counts = np.unique(dst_ports, return_counts=True)
                dst_port_entropy[second] = float(entropy(counts))
            else:
                dst_port_entropy[second] = 0.0
        else:
            src_port_entropy[second] = 0.0
            dst_port_entropy[second] = 0.0
    
    return src_port_entropy, dst_port_entropy

def main():
    """Enhanced main function with better UI, configuration management and error recovery"""
    print("\n" + "="*80)
    print("🔍 Network Traffic Feature Extraction Tool")
    print("="*80)
    
    try:
        while True:
            # Create separate Tk root for each file dialog to prevent issues
            root = Tk()
            root.title("Network Traffic Feature Extraction")
            root.geometry("300x100")  # Small window instead of fully hidden
            root.eval('tk::PlaceWindow . center')  # Center on screen
            
            print("\n📁 Select input files:")
            
            # File selection with clear instructions
            pcap_file = askopenfilename(
                title="Select PCAP file containing network traffic",
                filetypes=[("PCAP files", "*.pcap"), ("All files", "*.*")],
                parent=root
            )
            if not pcap_file:
                print("⚠️ No PCAP file selected. Exiting...")
                root.destroy()
                break
                
            print(f"✅ Selected PCAP: {os.path.basename(pcap_file)}")
            
            xml_file = askopenfilename(
                title="Select XML file containing attack labels",
                filetypes=[("XML files", "*.xml"), ("All files", "*.*")],
                parent=root
            )
            if not xml_file:
                print("⚠️ No XML file selected. Exiting...")
                root.destroy()
                break
                
            print(f"✅ Selected XML: {os.path.basename(xml_file)}")
            
            # Destroy root after file selection
            root.destroy()
            
            # Always extract all features
            attack_type = "any"
            
            # Data purpose selection with validation
            while True:
                train_or_test = input("\n📊 Enter 1 for training data or 2 for testing data: ").strip()
                if train_or_test in ["1", "2"]:
                    break
                print("❌ Invalid choice. Please enter 1 or 2.")
            
            # Determine output directory based on data purpose only
            if train_or_test == "1":
                output_directory = os.path.join(PROJECT_ROOT, "Phase 3", "Training")
                print(f"📂 Data will be saved to Training dataset")
            else:
                output_directory = os.path.join(PROJECT_ROOT, "Phase 3", "Testing")
                print(f"📂 Data will be saved to Testing dataset")
            
            # Create directory if it doesn't exist
            os.makedirs(output_directory, exist_ok=True)
            
            # Generate timestamp and paths
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(os.path.basename(pcap_file))[0]
            final_csv = os.path.join(output_directory, f"{base_name}_{timestamp}.csv")
            
            # Show selected options before processing
            print("\n📋 Processing Configuration:")
            print(f"  - Input PCAP: {pcap_file}")
            print(f"  - Input XML: {xml_file}")
            print(f"  - Output CSV: {final_csv}")
            print(f"  - Batch Size: {CONFIG['BATCH_SIZE']} packets")
            
            # Confirm before proceeding with long operation
            confirm = input("\n▶️ Proceed with processing? (Y/n): ").strip().lower()
            if confirm and confirm != "y":
                print("⚠️ Operation canceled by user")
                continue
                
            try:
                # Process the files with fixed "any" attack type to extract all features
                result = process_pcap_to_final(
                    pcap_file, xml_file, final_csv, timestamp, output_directory, "any"
                )
                
            except KeyboardInterrupt:
                print("\n⚠️ Processing interrupted by user")
            except Exception as e:
                print(f"\n❌ Unexpected error during processing: {e}")
                if CONFIG["DEBUG_MODE"]:
                    import traceback
                    traceback.print_exc()
            
            # Ask if user wants to process another file
            process_another = input("\n🔄 Process another file? (y/N): ").strip().lower() == "y"
            if not process_another:
                print("👋 Exiting...")
                break
                
    except KeyboardInterrupt:
        print("\n👋 Program terminated by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        if CONFIG.get("debug_mode", False):
            import traceback
            traceback.print_exc()
    finally:
        # Cleanup before exit
        print("\n✨ Processing complete. Cleaning up...")
        import gc
        gc.collect()

if __name__ == "__main__":
    try:            
        main()
    except KeyboardInterrupt:
        print("\n👋 Program terminated by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        if CONFIG["DEBUG_MODE"]:
            import traceback
            traceback.print_exc()
        sys.exit(1)