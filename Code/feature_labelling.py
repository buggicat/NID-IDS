import os
import json
import traceback
import ipaddress
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

# Common constants
ATTACK_TYPE_NAMES = {
    0: "normal",
    1: "ddosattack",
    2: "memcrashedspooferattack",
    3: "portscanattack",
}

# Utility functions
def ensure_directory(file_path):
    """Create directory if it doesn't exist."""
    output_dir = os.path.dirname(file_path)
    if output_dir and not os.path.exists(output_dir):
        print(f"   Creating directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
def save_dataframe(df, file_path):
    """Save dataframe to CSV with error handling."""
    ensure_directory(file_path)
    try:
        df.to_csv(file_path, index=False)
        print(f"✅ Data saved to {file_path}")
        print(f"   Output shape: {df.shape}")
        return True
    except PermissionError:
        print(f"❌ Permission denied when saving to {file_path}")
        return False
    except Exception as e:
        print(f"❌ Error saving data: {e}")
        return False

def print_attack_distribution(df):
    """Print attack distribution statistics."""
    if "attack" not in df.columns:
        return
        
    value_counts = df["attack"].value_counts().sort_index()
    print("   Attack labels distribution:")
    
    for label, count in value_counts.items():
        attack_name = ATTACK_TYPE_NAMES.get(label, f"unknown_attack_{label}")
        print(f"     - Label {label} ({attack_name}): {count} packets ({count/len(df)*100:.2f}%)")

def label_attack_data(csv_file, xml_file, output_csv, attack_type="any"):
    """Parse attack labels from XML file and add to dataset with support for multiple attack types."""
    try:
        # Validate input files exist before processing
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        if not os.path.exists(xml_file):
            raise FileNotFoundError(f"XML file not found: {xml_file}")
        
        # Read data
        try:
            df = pd.read_csv(csv_file)
        except pd.errors.EmptyDataError:
            print("❌ Empty CSV file - no data to process")
            return False
        except pd.errors.ParserError:
            print("❌ CSV parsing error - file may be corrupt")
            return False
            
        # Initialize attack columns
        df["attack"] = 0  
        df["attack_type"] = "normal"
        
        # Parse XML file
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # Find all attack elements
            attack_elements = root.findall(".//attack")
            
            if not attack_elements:
                print("⚠️ No attack elements found in XML file")
                return save_dataframe(df, output_csv)
                
            print(f"📊 Found {len(attack_elements)} attack definition(s) in XML file")
            
            # Process each attack element
            for idx, attack_elem in enumerate(attack_elements):
                attack_name_elem = attack_elem.find("./name")
                
                if attack_name_elem is None:
                    print(f"⚠️ Attack #{idx+1} has no name element, skipping")
                    continue
                    
                current_attack_name = attack_name_elem.text.lower() if attack_name_elem.text else f"unknown_attack_{idx}"
                print(f"  - Processing attack: {current_attack_name}")
                
                # Skip if we're filtering by attack_type and this doesn't match
                if attack_type != "any" and attack_type.lower() != current_attack_name:
                    print(f"  - Skipping {current_attack_name} as it doesn't match requested type {attack_type}")
                    continue
                
                # Get timestamp elements for this attack
                timestamp_start_elem = attack_elem.find(".//timestamp_start/timestamp")
                timestamp_end_elem = attack_elem.find(".//timestamp_end/timestamp")
                
                # Get attack parameters based on attack type
                attacker_ips_elem = None
                victim_ip_elem = None
                
                # Handle different attack types with specific parameter structures
                if "portscanattack" in current_attack_name.lower():
                    attacker_ips_elem = attack_elem.find(".//parameters/ip.src") or attack_elem.find(".//ip.src")
                    victim_ip_elem = attack_elem.find(".//parameters/ip.dst") or attack_elem.find(".//ip.dst")
                    
                elif "memcrashedspooferattack" in current_attack_name.lower():
                    attacker_ips_elem = attack_elem.find(".//parameters/ip.src") or attack_elem.find(".//ip.src")
                    victim_ip_elem = attack_elem.find(".//parameters/ip.victim") or attack_elem.find(".//ip.victim")
                    if victim_ip_elem is None:
                        victim_ip_elem = attack_elem.find(".//parameters/ip.dst") or attack_elem.find(".//ip.dst")
                    
                elif "ddosattack" in current_attack_name.lower():
                    attacker_ips_elem = attack_elem.find(".//parameters/ip.src") or attack_elem.find(".//ip.src")
                    victim_ip_elem = attack_elem.find(".//parameters/ip.dst") or attack_elem.find(".//ip.dst")
                    
                else:
                    # Generic approach for other attack types
                    attacker_ips_elem = attack_elem.find(".//parameters/ip.src") or attack_elem.find(".//ip.src")
                    victim_ip_elem = attack_elem.find(".//parameters/ip.dst") or attack_elem.find(".//ip.dst")
                
                if None in (timestamp_start_elem, timestamp_end_elem):
                    missing = []
                    if timestamp_start_elem is None: missing.append("timestamp_start")
                    if timestamp_end_elem is None: missing.append("timestamp_end")
                    print(f"⚠️ Attack '{current_attack_name}' missing required timestamp elements: {', '.join(missing)}, skipping")
                    continue
                
                # For some attacks like DDoS, the victim might be the attacker.ip.dst
                if victim_ip_elem is None and "ddosattack" in current_attack_name.lower():
                    print(f"⚠️ For DDoS attack, using destination IP as victim")
                    victim_ip_elem = attack_elem.find(".//ip.dst")
                    
                if None in (attacker_ips_elem, victim_ip_elem):
                    missing = []
                    if attacker_ips_elem is None: missing.append("ip.src")
                    if victim_ip_elem is None: missing.append("ip.dst/ip.victim")
                    print(f"⚠️ Attack '{current_attack_name}' missing required IP elements: {', '.join(missing)}, skipping")
                    continue
                    
                # Parse elements with safer default values
                try:
                    timestamp_start = float(timestamp_start_elem.text)
                    timestamp_end = float(timestamp_end_elem.text)
                except (ValueError, TypeError):
                    print(f"⚠️ Invalid timestamp format for attack '{current_attack_name}', skipping")
                    continue
                
                # Parse attacker IPs
                attacker_ips_raw = attacker_ips_elem.text if attacker_ips_elem is not None else None
                if attacker_ips_raw:
                    try:
                        # Handle list of IPs (common in DDoS attacks)
                        if attacker_ips_raw.startswith("[") and attacker_ips_raw.endswith("]"):
                            try:
                                # Try multiple parsing methods
                                try:
                                    # First try with proper quotes
                                    attacker_ips = json.loads(attacker_ips_raw.replace("'", "\""))
                                except json.JSONDecodeError:
                                    # Second try with ast.literal_eval for Python list syntax
                                    import ast
                                    attacker_ips = ast.literal_eval(attacker_ips_raw)
                                    
                                if not isinstance(attacker_ips, list):
                                    attacker_ips = [attacker_ips_raw]
                                    
                                print(f"    - Found {len(attacker_ips)} attacker IPs for {current_attack_name}")
                            except Exception as e:
                                print(f"⚠️ Invalid attacker IPs format: {e}")
                                print(f"⚠️ Raw value: {attacker_ips_raw}")
                                attacker_ips = [attacker_ips_raw]  # Use as single IP
                        else:
                            attacker_ips = [attacker_ips_raw]  # Assume single IP
                    except Exception as e:
                        print(f"⚠️ Error parsing attacker IPs: {e}")
                        continue
                else:
                    print("⚠️ No attacker IPs found")
                    attacker_ips = []
                    
                # Get victim IP
                victim_ip = victim_ip_elem.text
                
                # Ensure IP columns are strings for proper comparison
                if "ip.src" in df.columns and df["ip.src"].dtype != 'object':
                    df["ip.src"] = df["ip.src"].astype(str)
                if "ip.dst" in df.columns and df["ip.dst"].dtype != 'object':
                    df["ip.dst"] = df["ip.dst"].astype(str)
                
                # Add debug information
                print(f"    - Timestamp range: {timestamp_start} to {timestamp_end}")
                print(f"    - Attack type: {current_attack_name}")
                
                # Process IP formats
                if "ip.src" in df.columns:
                    if df["ip.src"].dtype != 'object':
                        print("   - IP columns are numeric - using numeric comparison")
                    else:
                        print("   - IP columns are strings - using string comparison")
                        
                        # Check if IPs look like integers
                        sample = df["ip.src"].iloc[0] if len(df) > 0 else ""
                        if sample and sample.isdigit():
                            print("   - Converting string IPs that look like integers to actual integers")
                            try:
                                df["ip.src"] = pd.to_numeric(df["ip.src"], errors="coerce")
                                if "ip.dst" in df.columns:
                                    df["ip.dst"] = pd.to_numeric(df["ip.dst"], errors="coerce")
                            except:
                                print("   - Failed to convert IPs to numeric, will use as strings")
                
                # Create attack mask based on attack type
                if "portscanattack" in current_attack_name.lower():
                    # Get expected packet count for verification
                    expected_count = int(attack_elem.find(".//injected_packets").text) if attack_elem.find(".//injected_packets") is not None else 0
                    print(f"    - Expected packet count from XML: {expected_count}")
                    
                    # Use the bidirectional traffic approach with a precise time window
                    attack_mask = (
                        (df["frame.time_epoch"] >= timestamp_start) &
                        (df["frame.time_epoch"] <= timestamp_end) &
                        (
                            # All outbound packets from attacker to victim 
                            ((df["ip.src"].isin(attacker_ips)) & (df["ip.dst"] == victim_ip)) |
                            
                            # All response packets from victim back to attacker
                            ((df["ip.src"] == victim_ip) & (df["ip.dst"].isin(attacker_ips)))
                        )
                    )
                    
                    # Use attack label 3 for port scan
                    attack_count = attack_mask.sum()
                    print(f"    - Initial PortScan detection: {attack_count} packets")
                    
                    # Refine detection if needed
                    if expected_count > 0 and (attack_count < expected_count * 0.95 or attack_count > expected_count * 1.05):
                        print(f"    - Refining PortScan detection to match expected count...")
                        
                        if "tcp.flags.syn" in df.columns:
                            # Port scans typically use SYN packets
                            syn_flag = (df["tcp.flags.syn"] == 1)
                            refined_mask = attack_mask & syn_flag
                            
                            if refined_mask.sum() > expected_count * 0.5:
                                attack_mask = refined_mask
                                print(f"    - Refined using SYN flags: {refined_mask.sum()} packets")
                                
                        # Try port-based refinement if needed
                        if abs(attack_mask.sum() - expected_count) > expected_count * 0.05:
                            if "dstport" in df.columns:
                                scan_window = df[attack_mask]
                                if len(scan_window) > 0:
                                    # Find ports that appear with high frequency
                                    port_counts = scan_window["dstport"].value_counts()
                                    significant_ports = port_counts[port_counts > 10].index.tolist()
                                    
                                    if significant_ports:
                                        print(f"    - Found {len(significant_ports)} significant destination ports")
                                        port_mask = df["dstport"].isin(significant_ports)
                                        
                                        time_mask = (df["frame.time_epoch"] >= timestamp_start) & (df["frame.time_epoch"] <= timestamp_end)
                                        ip_mask = ((df["ip.src"].isin(attacker_ips)) & (df["ip.dst"] == victim_ip))
                                        port_refined_mask = time_mask & ip_mask & port_mask
                                        
                                        if abs(port_refined_mask.sum() - expected_count) < abs(attack_mask.sum() - expected_count):
                                            attack_mask = port_refined_mask
                                            print(f"    - Refined using destination ports: {port_refined_mask.sum()} packets")
                    
                    # Apply the labels
                    df.loc[attack_mask, "attack"] = 3  # Using 3 for port scan
                    df.loc[attack_mask, "attack_type"] = current_attack_name
                    print(f"    - Final PortScan packets labeled: {attack_mask.sum()}")

                elif "memcrashedspooferattack" in current_attack_name.lower():
                    print(f"    - Processing Memcrashed attack with victim IP: {victim_ip}")
                    
                    # Simple time-based approach first
                    time_mask = (df["frame.time_epoch"] >= timestamp_start) & (df["frame.time_epoch"] <= timestamp_end)
                    
                    # Check for UDP protocol
                    udp_mask = pd.Series(True, index=df.index)
                    if "ip.proto" in df.columns:
                        udp_mask = (df["ip.proto"] == 17)  # 17 is UDP
                    elif "protocol" in df.columns:
                        udp_mask = (df["protocol"] == "UDP")
                    
                    # Look for packets involving the victim IP
                    victim_traffic = (df["ip.src"] == victim_ip) | (df["ip.dst"] == victim_ip)
                    
                    # Combine masks
                    attack_mask = time_mask & udp_mask & victim_traffic
                    
                    # Refine with memcached port if available
                    if "srcport" in df.columns or "dstport" in df.columns:
                        memcached_port_mask = False
                        if "srcport" in df.columns:
                            memcached_port_mask = memcached_port_mask | (df["srcport"] == 11211)
                        if "dstport" in df.columns:
                            memcached_port_mask = memcached_port_mask | (df["dstport"] == 11211)
                        
                        if isinstance(memcached_port_mask, pd.Series):
                            refined_mask = attack_mask & memcached_port_mask
                            if refined_mask.sum() > 0:
                                attack_mask = refined_mask
                                print(f"    - Using port-refined detection for Memcrashed ({refined_mask.sum()} packets)")
                    
                    # Apply Memcrashed attack label (2)
                    df.loc[attack_mask, "attack"] = 2
                    df.loc[attack_mask, "attack_type"] = current_attack_name
                    
                    # Log statistics
                    attack_count = attack_mask.sum()
                    print(f"    - Labeled {attack_count} packets as Memcrashed attack")
                    
                elif "ddosattack" in current_attack_name.lower():
                    print(f"    - Processing DDoS attack targeting {victim_ip}")
                    
                    # Get expected packet count for verification
                    expected_count = int(attack_elem.find(".//injected_packets").text) if attack_elem.find(".//injected_packets") is not None else 0
                    print(f"    - Expected packet count from XML: {expected_count}")
                    
                    # Use precise time window
                    time_mask = (
                        (df["frame.time_epoch"] >= timestamp_start) &
                        (df["frame.time_epoch"] <= timestamp_end)
                    )
                    
                    # Direct IP matching
                    attack_mask = pd.Series(False, index=df.index)
                    
                    if isinstance(attacker_ips, list) and len(attacker_ips) > 0:
                        src_match = df["ip.src"].isin(attacker_ips)
                        
                        if victim_ip:
                            dst_match = (df["ip.dst"] == victim_ip)
                            ip_mask = src_match & dst_match & time_mask
                        else:
                            ip_mask = src_match & time_mask
                            
                        attack_mask = ip_mask
                        
                        # Add responses if needed
                        if attack_mask.sum() < expected_count * 0.9:
                            response_mask = (df["ip.src"] == victim_ip) & (df["ip.dst"].isin(attacker_ips)) & time_mask
                            attack_mask = attack_mask | response_mask
                    
                    # Count how many packets were identified
                    attack_count = attack_mask.sum()
                    print(f"    - Initial DDoS detection: {attack_count} packets")
                    
                    # Refine if needed
                    if expected_count > 0 and abs(attack_count - expected_count) > expected_count * 0.05:
                        print(f"    - Refining DDoS detection to match expected count...")
                        
                        difference = expected_count - attack_count
                        
                        if difference > 0:
                            # Add more packets if needed
                            margin = 1.0  # 1 second margin
                            extended_time_mask = (
                                (df["frame.time_epoch"] >= timestamp_start - margin) &
                                (df["frame.time_epoch"] <= timestamp_end + margin)
                            ) & ~time_mask
                            
                            if isinstance(attacker_ips, list) and len(attacker_ips) > 0:
                                ip_mask = (df["ip.src"].isin(attacker_ips) | df["ip.dst"].isin(attacker_ips))
                                if victim_ip:
                                    ip_mask = ip_mask | (df["ip.src"] == victim_ip) | (df["ip.dst"] == victim_ip)
                                    
                                extended_mask = extended_time_mask & ip_mask & (df["attack"] == 0)
                                
                                if extended_mask.sum() > 0:
                                    candidates = df[extended_mask].index.tolist()
                                    to_add = min(len(candidates), difference)
                                    additional_packets = candidates[:to_add]
                                    
                                    attack_mask = attack_mask | df.index.isin(additional_packets)
                                    print(f"    - Added {to_add} packets from time margin")
                        
                        elif difference < 0:
                            # Remove excess packets
                            if attack_mask.sum() > 0:
                                attack_packets = df[attack_mask].copy()
                                
                                # Score packets by likelihood of being part of the attack
                                attack_packets["direct_flow"] = ((attack_packets["ip.src"].isin(attacker_ips)) & 
                                                                (attack_packets["ip.dst"] == victim_ip)).astype(int) * 3
                                                                
                                # Consider packet size
                                if "frame.len" in attack_packets.columns:
                                    max_size = attack_packets["frame.len"].max()
                                    if max_size > 0:
                                        attack_packets["size_score"] = 1 - (attack_packets["frame.len"] / max_size)
                                    else:
                                        attack_packets["size_score"] = 1
                                else:
                                    attack_packets["size_score"] = 1
                                
                                # Consider protocol
                                if "ip.proto" in attack_packets.columns:
                                    attack_packets["protocol_score"] = attack_packets["ip.proto"].isin([1, 17]).astype(int)
                                else:
                                    attack_packets["protocol_score"] = 1
                                
                                # Calculate total score
                                attack_packets["attack_score"] = attack_packets["direct_flow"] + attack_packets["size_score"] + attack_packets["protocol_score"]
                                
                                # Keep only the top packets by score
                                to_keep = attack_packets.sort_values("attack_score", ascending=False).head(expected_count).index
                                
                                # Create refined mask
                                refined_mask = df.index.isin(to_keep)
                                
                                # Update attack mask
                                attack_mask = refined_mask
                                print(f"    - Kept top {len(to_keep)} packets by attack likelihood score")
                    
                    # Label with value 1 for DDoS
                    df.loc[attack_mask, "attack"] = 1
                    df.loc[attack_mask, "attack_type"] = current_attack_name
                    
                    # Final count
                    final_count = attack_mask.sum()
                    print(f"    - Final DDoS packets labeled: {final_count} ({final_count/expected_count*100:.1f}% of expected)")
                else:
                    # Generic approach for other attack types
                    df.loc[attack_mask, "attack"] = 4  # Use higher numbers for other attack types
                    df.loc[attack_mask, "attack_type"] = current_attack_name
                
                # Log attack distribution
                attack_count = attack_mask.sum()
                print(f"    - Labeled {attack_count} packets as '{current_attack_name}'")
            
            # Summary statistics
            attack_count = df["attack"].sum()
            total_count = len(df)
            
            if total_count > 0:
                print(f"Labeled {attack_count} attack packets out of {total_count} total packets "
                    f"({attack_count/total_count*100:.2f}%)")
                
                attack_type_counts = df["attack_type"].value_counts()
                print("Attack type distribution:")
                for attack_type_name, count in attack_type_counts.items():
                    print(f"  - {attack_type_name}: {count} packets ({count/total_count*100:.2f}%)")
            
            # Save the processed data
            return save_dataframe(df, output_csv)
                
        except ET.ParseError:
            print("❌ XML parsing error - file may be corrupt")
            return False
            
    except Exception as e:
        print(f"❌ Error in label_attack_data: {str(e)}")
        traceback.print_exc()
        return False

def convert_to_numerical(csv_file, output_csv):
    """Convert categorical features to numerical for machine learning models."""
    try:
        df = pd.read_csv(csv_file)
        
        print(f"🔄 Converting categorical features to numerical...")
        print(f"   Input shape: {df.shape}")

        # Convert IP addresses
        for ip_col in ["ip.src", "ip.dst"]:
            if ip_col in df.columns and df[ip_col].dtype == 'object':
                print(f"   Converting {ip_col} addresses...")
                df[ip_col] = df[ip_col].apply(lambda ip: int(ipaddress.IPv4Address(ip)) if pd.notnull(ip) else 0)

        # Ensure IP protocol is numeric
        if "ip.proto" in df.columns:
            df["ip.proto"] = pd.to_numeric(df["ip.proto"], errors="coerce").fillna(0).astype(int)

        # Convert attack labels if needed
        if "attack" in df.columns:
            if df["attack"].dtype == 'object':
                attack_mapping = {
                    "normal": 0,
                    "ddosattack": 1,
                    "memcrashedspooferattack": 2,
                    "portscanattack": 3
                }
                
                print("   Mapping attack labels to specific numeric values")
                df["attack"] = df["attack"].map(attack_mapping).fillna(0).astype(int)
            else:
                print("   Attack column is already numeric, preserving values")
            
            # Print attack distribution
            print_attack_distribution(df)

        # Remove attack_type column (we already have numeric attack column)
        if "attack_type" in df.columns and df["attack_type"].dtype == 'object':
            print(f"   Removing attack_type column (using numeric attack column instead)")
            df = df.drop(columns=["attack_type"])

        # Convert port columns
        for port_col in ["srcport", "dstport"]:
            if port_col in df.columns and df[port_col].dtype == 'object':
                print(f"   Converting {port_col} to numeric")
                df[port_col] = pd.to_numeric(df[port_col], errors="coerce").fillna(0).astype(int)
            
        # Convert boolean columns
        bool_columns = [
            "is_udp", "is_memcached_port", "is_memcached_server", 
            "potential_spoofed_ip", "req_size_consistency", "potential_amplified",
            "tcp.flags.syn", "tcp.flags.ack", "tcp.flags.rst", "tcp.flags.fin"
        ]
        
        for col in bool_columns:
            if col in df.columns and df[col].dtype == 'object':
                print(f"     - Converting {col} to binary integer")
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

        # Convert any remaining string columns
        string_cols = df.select_dtypes(include=['object']).columns
        string_cols = [col for col in string_cols if col != "attack_type"]
        
        if len(string_cols) > 0:
            print(f"   Converting {len(string_cols)} additional string columns to numeric")
            for col in string_cols:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                except:
                    print(f"⚠️ Couldn't convert column {col}, dropping it")
                    df = df.drop(columns=[col])

        # Save the processed data
        return save_dataframe(df, output_csv)
            
    except Exception as e:
        print(f"❌ Error in convert_to_numerical: {e}")
        traceback.print_exc()
        return False

def normalize_data(csv_file, output_csv):
    """Normalize data to the 0-1 range for better model performance."""
    try:
        print("🔄 Loading data for normalization...")
        df = pd.read_csv(csv_file)
        print(f"   Input shape: {df.shape}")
        
        # Preserve attack column before normalization
        if "attack" in df.columns:
            print("   Preserving attack column values for classification")
            attack_values = df["attack"].copy()
        
        # Exclude non-feature columns
        exclude_columns = ["attack", "frame.time_epoch", "attack_type", "attack_type_numeric"]
        
        # Remove attack_type columns
        for col in ["attack_type", "attack_type_numeric"]:
            if col in df.columns:
                print(f"   Removing '{col}' column")
                df = df.drop(columns=[col])
            
        # Identify columns to normalize
        columns_to_normalize = [col for col in df.columns if col not in exclude_columns]
        existing_columns = [col for col in columns_to_normalize if col in df.columns]
        
        if not existing_columns:
            print("⚠️ No columns to normalize were found in the dataset")
            return save_dataframe(df, output_csv)
        
        print(f"🔄 Normalizing {len(existing_columns)} numerical features...")
        
        if len(df) > 1:
            # Convert any object columns to numeric first
            for col in existing_columns:
                if df[col].dtype == 'object':
                    print(f"   Converting column {col} to numeric for normalization")
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            
            # Handle infinities and NaN values
            print("   Cleaning data before normalization...")
            df[existing_columns] = df[existing_columns].replace([np.inf, -np.inf], np.nan)
            
            # Normalize using MinMaxScaler
            try:
                scaler = MinMaxScaler()
                df[existing_columns] = scaler.fit_transform(df[existing_columns])
                print("✅ Data normalized using MinMaxScaler")
                
                # Verify values are within range
                min_values = df[existing_columns].min().min()
                max_values = df[existing_columns].max().max()
                
                if (min_values < 0 or max_values > 1):
                    print(f"⚠️ Warning: Some normalized values outside [0,1] range: min={min_values:.4f}, max={max_values:.4f}")
                    print("   Clipping values to [0,1] range")
                    df[existing_columns] = df[existing_columns].clip(0, 1)
            except Exception as e:
                print(f"❌ Error during normalization: {e}")
                print("   Proceeding with original values")
                traceback.print_exc()
            
            # Handle any remaining NaN values
            nan_count = df[existing_columns].isna().sum().sum()
            if nan_count > 0:
                print(f"⚠️ Found {nan_count} NaN values after normalization, filling with zeros")
                df[existing_columns] = df[existing_columns].fillna(0)
        else:
            print("⚠️ Warning: Not enough data for proper normalization")
        
        # Restore attack column if it existed
        if "attack" in df.columns and "attack_values" in locals():
            print("   Restoring original attack column values")
            df["attack"] = attack_values
            print_attack_distribution(df)
        
        # Filter features based on feature_dict.json
        print("\n🔄 Filtering features based on feature_dict.json...")
        feature_dict_path = os.path.join(os.path.dirname(__file__), 'feature_dict.json')
        
        if os.path.exists(feature_dict_path):
            try:
                with open(feature_dict_path, 'r') as f:
                    allowed_features = json.load(f)
                
                # Ensure attack column is in allowed features
                if "attack" in df.columns and "attack" not in allowed_features:
                    allowed_features.append("attack")
                    
                print(f"   Keeping only {len(allowed_features)} features from feature_dict.json")
                
                # Determine which features to drop
                features_to_drop = [col for col in df.columns if col not in allowed_features]
                
                if features_to_drop:
                    print(f"   Removing {len(features_to_drop)} features not in feature_dict.json:")
                    for feature in features_to_drop[:10]:
                        print(f"     - {feature}")
                    if len(features_to_drop) > 10:
                        print(f"     - ...and {len(features_to_drop) - 10} more")
                    
                    # Remove unwanted features
                    df = df.drop(columns=features_to_drop, errors='ignore')
                
                # Check for features in JSON but missing from data
                missing_features = [f for f in allowed_features if f not in df.columns]
                if missing_features:
                    print(f"⚠️ Warning: {len(missing_features)} features in feature_dict.json are not present in dataset:")
                    for feature in missing_features[:5]:
                        print(f"     - {feature}")
                    if len(missing_features) > 5:
                        print(f"     - ...and {len(missing_features) - 5} more")
                
                print(f"✅ Final dataset contains {len(df.columns)} features")
            except Exception as e:
                print(f"⚠️ Warning: Could not filter features using feature_dict.json: {e}")
                print(f"   Proceeding with current feature set")
                traceback.print_exc()
        else:
            print(f"⚠️ Warning: feature_dict.json not found at {feature_dict_path}")
            print(f"   Proceeding with current feature set")
        
        # Save the processed data
        return save_dataframe(df, output_csv)
            
    except Exception as e:
        print(f"❌ Error in normalize_data: {e}")
        traceback.print_exc()
        return False

def main(input_csv, xml_file=None, output_csv=None, normalize=True, attack_type="any"):
    """
    Main entry point for the feature labelling pipeline.
    
    Args:
        input_csv (str): Path to the input CSV file with network traffic data
        xml_file (str, optional): Path to XML file with attack labels
        output_csv (str, optional): Path to save the final output CSV file
        normalize (bool): Whether to normalize the features
        attack_type (str): Type of attack to filter for ("any" for all attacks)
        
    Returns:
        tuple: (success, output_path) where:
            - success (bool): Whether the processing was successful
            - output_path (str): Path to the final output file
    """
    try:
        if not os.path.exists(input_csv):
            print(f"❌ Input CSV file not found: {input_csv}")
            return False, None
        
        # Set default output path if not provided
        if output_csv is None:
            base_dir = os.path.dirname(input_csv)
            base_name = os.path.splitext(os.path.basename(input_csv))[0]
            output_csv = os.path.join(base_dir, f"{base_name}_processed.csv")
        
        # Create paths for intermediate files
        temp_dir = os.path.dirname(output_csv) or os.path.dirname(input_csv) or "."
        base_name = os.path.splitext(os.path.basename(output_csv))[0]
        labeled_csv = os.path.join(temp_dir, f"{base_name}_labeled.csv")
        numerical_csv = os.path.join(temp_dir, f"{base_name}_numerical.csv")
        
        # Track current working file
        current_file = input_csv
        
        # Step 1: Label data if XML file provided
        if xml_file:
            print(f"🔍 Step 1/3: Labelling attack data from {xml_file}")
            if not os.path.exists(xml_file):
                print(f"❌ XML file not found: {xml_file}")
                return False, None
                
            success = label_attack_data(current_file, xml_file, labeled_csv, attack_type)
            if not success:
                print(f"❌ Failed to label attack data")
                return False, None
            current_file = labeled_csv
        else:
            print(f"🔍 Step 1/3: Skipping labelling (no XML file provided)")
        
        # Step 2: Convert categorical features to numerical
        print(f"🔍 Step 2/3: Converting categorical features to numerical")
        success = convert_to_numerical(current_file, numerical_csv)
        if not success:
            print(f"❌ Failed to convert to numerical data")
            return False, None
        current_file = numerical_csv
        
        # Step 3: Normalize and filter features
        if normalize:
            print(f"🔍 Step 3/3: Normalizing features and applying feature filtering")
            success = normalize_data(current_file, output_csv)
            if not success:
                print(f"❌ Failed to normalize data")
                return False, None
        else:
            print(f"🔍 Step 3/3: Skipping normalization (normalize=False)")
            import shutil
            shutil.copy(current_file, output_csv)
        
        # Clean up temporary files
        temp_files = [labeled_csv, numerical_csv]
        for temp_file in temp_files:
            if temp_file != input_csv and temp_file != output_csv and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    print(f"🧹 Removed temporary file: {temp_file}")
                except:
                    pass
                    
        print(f"✅ Processing complete! Final output saved to: {output_csv}")
        return True, output_csv
        
    except Exception as e:
        print(f"❌ Error in main processing: {str(e)}")
        traceback.print_exc()
        return False, None

# Import message for when the script is imported
if __name__ == "__main__":
    print("This script is designed to be imported and called from another script.")
    print("Example usage:")
    print("  from feature_labelling import main")
    print("  success, output_file = main('input.csv', 'attacks.xml', 'output.csv')")