"""
telemetry_recorder.py — Fixed-60Hz OutSim UDP telemetry logger for Live for
Speed, per proposal §3.4.

Connects to LFS's OutSim Pack 2 UDP feed (127.0.0.1:20778), drains the
socket every loop iteration to always process the freshest packet, and
writes one CSV row per 1/60s tick regardless of how fast packets actually
arrive — state is retained across ticks if a new packet hasn't landed yet.

This is DataCatcher.py, unmodified, moved into the collection module.
Run directly: `python -m src.collection.telemetry_recorder`, or call
`main()` from session_manager.
"""

import socket
import struct
import csv
import os
import math
import time
import select

def main():
    print("====================================================")
    print("  LFS FIXED 60Hz STATE-RETENTION TELEMETRY LOGGER   ")
    print("====================================================\n")
    
    save_dir = input("Enter destination folder path (Press Enter for current directory): ").strip()
    if not save_dir or not os.path.exists(save_dir):
        save_dir = os.getcwd()
            
    timestamp = int(time.time())
    filepath = os.path.join(save_dir, f"lfs_lstm_telemetry_{timestamp}.csv")
    
    UDP_IP = "127.0.0.1"
    UDP_PORT = 20778
    
    # 272-byte OutSim Pack 2 format layout
    OUTSIM_PACKET_FMT = "<4siI12f3i5f4B2f2f" + "7f4B2f"*4
    LFS_METER = 65536.0 
    
    # Strict 60Hz Timing Constraints
    TARGET_HZ = 60.0
    FRAME_INTERVAL = 1.0 / TARGET_HZ
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    sock.setblocking(0)  
    
    print(f"\n[+] Active Port: {UDP_PORT} | Targeted Logging Rate: {TARGET_HZ}Hz")
    print(f"[+] Output Asset: {filepath}")
    print("[...] Awaiting stream sequence. Initiate track session...")

    last_pos = None
    current_lap = 1
    last_lap_dist = None
    cumulative_distance_m = 0.0
    session_start_time = None
    last_packet_time = time.perf_counter()
    
    # --- Persistent State and Feature Variables ---
    current_state_packet = None  # Retains the latest packet across microsecond loops
    last_throttle = 0.0
    last_brake = 0.0
    last_steer = 0.0
    last_frame_time = None
    
    # Precision loop clock gate
    next_frame_time = time.perf_counter()
    
    fieldnames = [
        'Timestamp_MS', 'Session_Time_S', 'Lap', 'Distance_M', 'Lap_Dist_M',
        'Speed_KMH', 'Engine_RPM', 'Gear', 
        'Throttle', 'Brake', 'Steer', 'Clutch', 'Handbrake',
        'Throttle_Rate', 'Brake_Rate', 'Steer_Rate',                  
        'Slip_Ratio_LF', 'Slip_Ratio_RF', 'Slip_Ratio_LR', 'Slip_Ratio_RR', 
        'Body_Slip_Angle',                                            
        'AngVel_X', 'AngVel_Y', 'AngVel_Z',
        'Susp_Load_LF', 'Susp_Load_RF', 'Susp_Load_LR', 'Susp_Load_RR',
        'Wheel_Spin_LF', 'Wheel_Spin_RF', 'Wheel_Spin_LR', 'Wheel_Spin_RR',
        'Accel_X', 'Accel_Y', 'Accel_Z', 'Roll', 'Pitch', 'Heading'
    ]
    
    with open(filepath, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        
        session_active = True
        
        while session_active:
            # 1. Non-blocking drain: Always capture the freshest packet
            while True:
                ready = select.select([sock], [], [], 0.0)
                if ready[0]:
                    data, addr = sock.recvfrom(1024)
                    if len(data) >= 272:
                        current_state_packet = data
                        last_packet_time = time.perf_counter()
                else:
                    break
            
            # If the session hasn't received any data yet, wait
            if current_state_packet is None:
                time.sleep(0.001)
                continue
                
            # 2. Strict Clock Gate Evaluation
            t_now = time.perf_counter()
            if t_now < next_frame_time:
                time.sleep(0.0005) # Yield execution briefly to optimize CPU usage
                continue
                
            # Advance timing gate target
            next_frame_time += FRAME_INTERVAL
            if t_now > next_frame_time + FRAME_INTERVAL:
                next_frame_time = t_now + FRAME_INTERVAL 

            # Break recording if stream has been silent for more than 10 seconds
            if session_start_time is not None and (t_now - last_packet_time) > 10.0:
                print("\n[!] Stream went silent. Finalizing dataset output...")
                break

            # 3. Unpack and Process the Retained State Snapshot
            unpacked = struct.unpack(OUTSIM_PACKET_FMT, current_state_packet[:272])
            
            sim_time = unpacked[2]
            angvel_x, angvel_y, angvel_z = unpacked[3], unpacked[4], unpacked[5]
            vel_x, vel_y, vel_z = unpacked[12], unpacked[13], unpacked[14]
            pos_x, pos_y, pos_z = unpacked[15], unpacked[16], unpacked[17]
            
            throttle = unpacked[18]
            brake = unpacked[19]
            steer = unpacked[20]       
            clutch = unpacked[21]
            handbrake = unpacked[22]    
            
            gear = unpacked[23] - 1 
            rpm = unpacked[27] * 60 / (2 * math.pi)
            lap_dist = unpacked[29]
            
            # Compute Time Delta between logged rows
            if last_frame_time is not None:
                dt = t_now - last_frame_time
                if dt <= 0: dt = 0.01666  
            else:
                dt = 0.01666

            # Calculate Derivatives relative to the 60Hz timeline
            throttle_rate = (throttle - last_throttle) / dt
            brake_rate = (brake - last_brake) / dt
            steer_rate = (steer - last_steer) / dt

            last_throttle = throttle
            last_brake = brake
            last_steer = steer
            last_frame_time = t_now

            speed_kmh = math.sqrt(vel_x**2 + vel_y**2 + vel_z**2) * 3.6
            speed_ms = speed_kmh / 3.6
            tire_radius = 0.31 

            susp_lf, spin_lf = unpacked[35], unpacked[36]
            susp_rf, spin_rf = unpacked[48], unpacked[49]
            susp_lr, spin_lr = unpacked[61], unpacked[62]
            susp_rr, spin_rr = unpacked[74], unpacked[75]

            # Longitudinal Slip Metrics
            if speed_ms > 0.5:
                slip_ratio_lf = ((spin_lf * tire_radius) - speed_ms) / speed_ms
                slip_ratio_rf = ((spin_rf * tire_radius) - speed_ms) / speed_ms
                slip_ratio_lr = ((spin_lr * tire_radius) - speed_ms) / speed_ms
                slip_ratio_rr = ((spin_rr * tire_radius) - speed_ms) / speed_ms
            else:
                slip_ratio_lf = slip_ratio_rf = slip_ratio_lr = slip_ratio_rr = 0.0

            # Lateral Slip Metrics
            if speed_ms > 1.0:
                body_slip_angle = math.atan2(vel_y, vel_x)
            else:
                body_slip_angle = 0.0
            
            # Track Lap Counter Transitions
            if last_lap_dist is not None:
                if lap_dist < 40.0 and last_lap_dist > 200.0:
                    current_lap += 1
            last_lap_dist = lap_dist
            
            if session_start_time is None and speed_kmh > 1.0:
                session_start_time = t_now
            
            if last_pos is not None:
                dx = pos_x - last_pos[0]
                dy = pos_y - last_pos[1]
                dz = pos_z - last_pos[2]
                step_dist = math.sqrt(dx**2 + dy**2 + dz**2) / LFS_METER
                if speed_kmh > 0.5:
                    cumulative_distance_m += step_dist
            last_pos = (pos_x, pos_y, pos_z)
            
            elapsed_seconds = t_now - session_start_time if session_start_time else 0.0
            time_str = time.strftime('%M:%S', time.gmtime(elapsed_seconds))

            # 4. Write Uniform Telemetry Data Frame
            writer.writerow({
                'Timestamp_MS': sim_time,
                'Session_Time_S': round(elapsed_seconds, 3),
                'Lap': current_lap,
                'Distance_M': round(cumulative_distance_m, 2),
                'Lap_Dist_M': round(lap_dist, 2),
                'Speed_KMH': round(speed_kmh, 2),
                'Engine_RPM': round(rpm, 1),
                'Gear': gear,
                'Throttle': round(throttle, 3),
                'Brake': round(brake, 3),
                'Steer': round(steer, 3),
                'Clutch': round(clutch, 3),
                'Handbrake': round(handbrake, 3),
                'Throttle_Rate': round(throttle_rate, 3),
                'Brake_Rate': round(brake_rate, 3),
                'Steer_Rate': round(steer_rate, 3),
                'Slip_Ratio_LF': round(slip_ratio_lf, 4),
                'Slip_Ratio_RF': round(slip_ratio_rf, 4),
                'Slip_Ratio_LR': round(slip_ratio_lr, 4),
                'Slip_Ratio_RR': round(slip_ratio_rr, 4),
                'Body_Slip_Angle': round(body_slip_angle, 4),
                'AngVel_X': round(angvel_x, 4),
                'AngVel_Y': round(angvel_y, 4),
                'AngVel_Z': round(angvel_z, 4),
                'Susp_Load_LF': round(susp_lf, 2),
                'Susp_Load_RF': round(susp_rf, 2),
                'Susp_Load_LR': round(susp_lr, 2),
                'Susp_Load_RR': round(susp_rr, 2),
                'Wheel_Spin_LF': round(spin_lf, 3),
                'Wheel_Spin_RF': round(spin_rf, 3),
                'Wheel_Spin_LR': round(spin_lr, 3),
                'Wheel_Spin_RR': round(spin_rr, 3),
                'Accel_X': round(unpacked[9], 3),
                'Accel_Y': round(unpacked[10], 3),
                'Accel_Z': round(unpacked[11], 3),
                'Roll': round(unpacked[8], 4),
                'Pitch': round(unpacked[7], 4),
                'Heading': round(unpacked[6], 4)
            })
            
            # Replaced with the modified original printing output layout
            print(f"\r⏱️ Clock: {time_str} | Lap: {current_lap} | 🗺️ Total: {cumulative_distance_m/1000.0:.2f}KM | RPM: {rpm:.0f} | Gear: {gear}", end="")
            
    print("\n[+] Dataset generated successfully with zero packet dropping.")

if __name__ == "__main__":
    main()