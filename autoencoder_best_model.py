import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.signal import find_peaks
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import glob

mp_pose = mp.solutions.pose

RUNNING_LANDMARKS = {
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_elbow': 13,
    'right_elbow': 14,
    'left_wrist': 15,
    'right_wrist': 16,
    'left_hip': 23,
    'right_hip': 24,
    'left_knee': 25,
    'right_knee': 26,
    'left_ankle': 27,
    'right_ankle': 28,
    'left_heel': 29,
    'right_heel': 30,
    'left_foot_index': 31,
    'right_foot_index': 32
}

# --- Angle Calculation ---
def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# --- Pose Feature Extraction ---
def extract_pose_features(landmarks):
    features = {}
    pose_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])

    features['left_ankle_x'] = pose_array[RUNNING_LANDMARKS['left_ankle']][0]
    features['left_ankle_y'] = pose_array[RUNNING_LANDMARKS['left_ankle']][1]
    features['right_ankle_x'] = pose_array[RUNNING_LANDMARKS['right_ankle']][0]
    features['right_ankle_y'] = pose_array[RUNNING_LANDMARKS['right_ankle']][1]

    features['left_knee_angle'] = calculate_angle(
        pose_array[RUNNING_LANDMARKS['left_hip']][:2],
        pose_array[RUNNING_LANDMARKS['left_knee']][:2],
        pose_array[RUNNING_LANDMARKS['left_ankle']][:2]
    )
    features['right_knee_angle'] = calculate_angle(
        pose_array[RUNNING_LANDMARKS['right_hip']][:2],
        pose_array[RUNNING_LANDMARKS['right_knee']][:2],
        pose_array[RUNNING_LANDMARKS['right_ankle']][:2]
    )
    features['left_elbow_angle'] = calculate_angle(
        pose_array[RUNNING_LANDMARKS['left_shoulder']][:2],
        pose_array[RUNNING_LANDMARKS['left_elbow']][:2],
        pose_array[RUNNING_LANDMARKS['left_wrist']][:2]
    )
    features['right_elbow_angle'] = calculate_angle(
        pose_array[RUNNING_LANDMARKS['right_shoulder']][:2],
        pose_array[RUNNING_LANDMARKS['right_elbow']][:2],
        pose_array[RUNNING_LANDMARKS['right_wrist']][:2]
    )
    features['left_hip_angle'] = calculate_angle(
        pose_array[RUNNING_LANDMARKS['left_shoulder']][:2],
        pose_array[RUNNING_LANDMARKS['left_hip']][:2],
        pose_array[RUNNING_LANDMARKS['left_knee']][:2]
    )
    features['right_hip_angle'] = calculate_angle(
        pose_array[RUNNING_LANDMARKS['right_shoulder']][:2],
        pose_array[RUNNING_LANDMARKS['right_hip']][:2],
        pose_array[RUNNING_LANDMARKS['right_knee']][:2]
    )

    shoulder_center = (pose_array[RUNNING_LANDMARKS['left_shoulder']] + pose_array[RUNNING_LANDMARKS['right_shoulder']]) / 2
    hip_center = (pose_array[RUNNING_LANDMARKS['left_hip']] + pose_array[RUNNING_LANDMARKS['right_hip']]) / 2
    trunk_vector = shoulder_center - hip_center
    features['trunk_lean'] = np.degrees(np.arctan2(trunk_vector[0], trunk_vector[1]))

    features['stride_width'] = abs(
        pose_array[RUNNING_LANDMARKS['left_ankle']][0] - pose_array[RUNNING_LANDMARKS['right_ankle']][0]
    )
    features['center_of_mass_y'] = (hip_center[1] + shoulder_center[1]) / 2

    left_arm_swing = np.linalg.norm(pose_array[RUNNING_LANDMARKS['left_wrist']] - pose_array[RUNNING_LANDMARKS['left_shoulder']])
    right_arm_swing = np.linalg.norm(pose_array[RUNNING_LANDMARKS['right_wrist']] - pose_array[RUNNING_LANDMARKS['right_shoulder']])
    features['arm_swing_ratio'] = min(left_arm_swing, right_arm_swing) / max(left_arm_swing, right_arm_swing)

    return features

# --- Pose Cycle Extraction Utilities ---
def detect_gait_cycles(pose_data):
    left_ankle_y = pose_data['left_ankle_y'].values
    right_ankle_y = pose_data['right_ankle_y'].values

    left_strikes, _ = find_peaks(-left_ankle_y, distance=10, prominence=0.02)
    right_strikes, _ = find_peaks(-right_ankle_y, distance=10, prominence=0.02)

    gait_cycles = []
    for i in range(len(left_strikes) - 1):
        gait_cycles.append({
            'start_frame': left_strikes[i],
            'end_frame': left_strikes[i + 1],
            'foot': 'left',
            'duration_frames': left_strikes[i + 1] - left_strikes[i]
        })

    for i in range(len(right_strikes) - 1):
        gait_cycles.append({
            'start_frame': right_strikes[i],
            'end_frame': right_strikes[i + 1],
            'foot': 'right',
            'duration_frames': right_strikes[i + 1] - right_strikes[i]
        })

    return gait_cycles, left_strikes, right_strikes

def extract_single_gait_cycle(pose_data, cycle_info):
    start = cycle_info['start_frame']
    end = cycle_info['end_frame']
    cycle_data = pose_data.iloc[start:end].copy()
    cycle_data['cycle_progress'] = np.linspace(0, 100, len(cycle_data))
    return cycle_data

def normalize_gait_cycle(cycle_df, num_points=100):
    if len(cycle_df) < 2:
        return pd.DataFrame()  # ไม่สามารถ interpolate ได้

    x_old = np.linspace(0, 1, len(cycle_df))
    x_new = np.linspace(0, 1, num_points)
    interpolated = {}

    for col in cycle_df.columns:
        if cycle_df[col].dtype in [np.float32, np.float64, np.int64, np.int32]:
            interpolated[col] = np.interp(x_new, x_old, cycle_df[col])

    return pd.DataFrame(interpolated)

def process_video(video_path, sample_rate=3):
    """Process video and extract pose data"""
    print(f"Processing video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video info: {fps} FPS, {total_frames} frames")

    pose_data = []
    frame_count = 0

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process every nth frame based on sample_rate
            if frame_count % sample_rate == 0:
                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                if results.pose_landmarks:
                    features = extract_pose_features(results.pose_landmarks)
                    features['frame'] = frame_count
                    features['time'] = frame_count / fps
                    pose_data.append(features)

            frame_count += 1

    cap.release()
    print(f"✓ Processing complete! Extracted {len(pose_data)} poses")

    return pd.DataFrame(pose_data)

def convert_all_joints_to_leading_trailing(cycle_data, strike_foot):
    joints = set()
    for col in cycle_data.columns:
        if col.startswith('left_'):
            joints.add(col.replace('left_', '').rsplit('_', 1)[0])
        elif col.startswith('right_'):
            joints.add(col.replace('right_', '').rsplit('_', 1)[0])

    for joint in joints:
        for axis in ['x', 'y', 'z','angle']:  # เผื่อมี z-axis หรือ angle ด้วย
            left_col = f'left_{joint}_{axis}'
            right_col = f'right_{joint}_{axis}'
            leading_col = f'leading_{joint}_{axis}'
            trailing_col = f'trailing_{joint}_{axis}'
            if left_col in cycle_data.columns and right_col in cycle_data.columns:
                if strike_foot == 'left':
                    cycle_data[leading_col] = cycle_data[left_col]
                    cycle_data[trailing_col] = cycle_data[right_col]
                else:
                    cycle_data[leading_col] = cycle_data[right_col]
                    cycle_data[trailing_col] = cycle_data[left_col]

    return cycle_data

def process_and_label_gait_cycles(pose_data):
    gait_cycles, left_strikes, right_strikes = detect_gait_cycles(pose_data)
    all_cycles = []

    for cycle_info in gait_cycles:
        cycle_data = extract_single_gait_cycle(pose_data, cycle_info)
        cycle_data = convert_all_joints_to_leading_trailing(cycle_data, cycle_info['foot'])
        all_cycles.append(cycle_data)

    labeled_pose_data = pd.concat(all_cycles, ignore_index=True)

    cols_to_drop = [col for col in labeled_pose_data.columns if col.startswith('left_') or col.startswith('right_')]
    labeled_pose_data = labeled_pose_data.drop(columns=cols_to_drop)

    return labeled_pose_data, gait_cycles, left_strikes, right_strikes

# --- Autoencoder Definition ---
class GaitAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size=128, bottleneck_size=32):
        super(GaitAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, bottleneck_size),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed

# --- Dataset Class ---
class GaitCycleDataset(Dataset):
    def __init__(self, cycle_array_list):
        self.samples = np.stack(cycle_array_list)  # Ensures uniform shape

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx], dtype=torch.float32)

# --- Load Gait Cycles from Elite Videos ---
def load_gait_cycles_from_videos(video_paths, selected_features, num_points=100):
    all_cycles = []
    for path in video_paths:
        print(f"\n📹 Processing elite video: {path}")
        pose_data = process_video(path)
        labeled_pose_data, gait_cycles, left_strikes, right_strikes = process_and_label_gait_cycles(pose_data)
        print(f"  Extracted {len(pose_data)} poses from video")

        mid_cycle_idx = len(gait_cycles) // 2
        # selected_cycle = gait_cycles[mid_cycle_idx]
        # cycle_data = extract_single_gait_cycle(labeled_pose_data, selected_cycle)
        # normalized_cycle = normalize_gait_cycle(cycle_data)
        print(f"  Found {len(gait_cycles)} gait cycles")

        for cycle in gait_cycles:
            cycle_data = extract_single_gait_cycle(labeled_pose_data, cycle)
            normalized = normalize_gait_cycle(cycle_data, num_points)

            if all(f in normalized.columns for f in selected_features) and len(normalized) == num_points:
                sequence = normalized[selected_features].values
                all_cycles.append(sequence)

    print(f"\n✅ Loaded {len(all_cycles)} gait cycles total")
    return all_cycles

# --- Training Function ---
def train_gait_autoencoder(model, train_dataset, val_dataset=None, num_epochs=50, batch_size=16, learning_rate=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            B, T, F = batch.shape
            batch_flat = batch.view(B * T, F)

            output = model(batch_flat)
            loss = criterion(output, batch_flat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.6f}", end='')

        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    B, T, F = batch.shape
                    batch_flat = batch.view(B * T, F)
                    output = model(batch_flat)
                    loss = criterion(output, batch_flat)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            print(f" | Val Loss: {avg_val_loss:.6f}")
        else:
            print()


# --- Evaluation Function ---
def evaluate_gait_anomaly(model, pose_data, selected_features, num_points=100):
    model.eval()
    device = next(model.parameters()).device
    cycles, _, _ = detect_gait_cycles(pose_data)

    if not cycles:
        return None, "No gait cycles found"

    cycle = extract_single_gait_cycle(pose_data, cycles[0])
    norm_cycle = normalize_gait_cycle(cycle, num_points)

    if not all(f in norm_cycle.columns for f in selected_features):
        return None, "Missing features in pose data"

    x = torch.tensor(norm_cycle[selected_features].values, dtype=torch.float32).to(device)
    # x_flat = x.view(-1, x.shape[-1])
    x_flat = x.unsqueeze(0)  # add batch dimension: [1, T, F]
    recon = model(x_flat)

    print(f'x: {x}')
    print(f'x_flat: {x_flat}')
    print(f'recon: {recon}')

    with torch.no_grad():
        recon = model(x_flat)
        print(f'recon: {recon}')
        mse = torch.mean((x_flat - recon) ** 2, dim=1)
        mape = torch.mean(torch.abs((x_flat - recon) / (x_flat + 1e-8)), dim=1) * 100
        avg_error = mse.mean().item()
        avg_mape = mape.mean().item()

    return avg_error, mse.cpu().numpy(), avg_mape, mape.cpu().numpy()

# --- Process Video with MediaPipe ---
def process_video(video_path, sample_rate=3):
    print(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video info: {fps} FPS, {total_frames} frames")

    pose_data = []
    frame_count = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % sample_rate == 0:
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                if results.pose_landmarks:
                    features = extract_pose_features(results.pose_landmarks)
                    features['frame'] = frame_count
                    features['time'] = frame_count / fps
                    pose_data.append(features)

                    if len(pose_data) % 10 == 0:
                        print(f"Processed {len(pose_data)} poses...")

            frame_count += 1

    cap.release()
    print(f"✓ Processing complete! Extracted {len(pose_data)} poses")
    return pd.DataFrame(pose_data)

# --- RNN Model Factory ---
def create_model(input_size, encoder_type, decoder_type, hidden_size=256, bottleneck_size=64, num_layers=4):
    class Seq2SeqAutoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            rnn_class = getattr(nn, encoder_type)
            self.encoder_rnn = rnn_class(input_size, hidden_size, num_layers=num_layers, batch_first=True)
            self.bottleneck = nn.Linear(hidden_size, bottleneck_size)
            self.decoder_rnn = rnn_class(bottleneck_size, hidden_size, num_layers=num_layers, batch_first=True)
            self.output_layer = nn.Linear(hidden_size, input_size)

        def forward(self, x):
            if encoder_type == 'LSTM':
                _, (h_n, _) = self.encoder_rnn(x)
            else:
                _, h_n = self.encoder_rnn(x)

            latent = self.bottleneck(h_n[-1])
            z = latent.unsqueeze(1).repeat(1, x.size(1), 1)
            decoded, _ = self.decoder_rnn(z)
            return self.output_layer(decoded)

    return Seq2SeqAutoencoder()

def train_model_with_validation(model, train_dataset, val_dataset, model_name, num_epochs=50, batch_size=16, learning_rate=1e-3, patience=5, feature=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss = float('inf')
    best_val_mape = float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            B, T, F = batch.shape
            output = model(batch)
            loss = criterion(output.view(B*T, F), batch.view(B*T, F))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        val_mape = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                B, T, F = batch.shape
                output = model(batch)
                loss = criterion(output.view(B*T, F), batch.view(B*T, F))
                val_loss += loss.item()

                # Mean Absolute Percentage Error
                abs_perc_err = torch.abs((batch - output) / (batch + 1e-8)) * 100
                mape = abs_perc_err.mean().item()
                val_mape += mape

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_mape = val_mape / len(val_loader)

        print(f"{model_name} - Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val MAPE: {avg_val_mape:.2f}%")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_mape = avg_val_mape
            torch.save(model.state_dict(), f"best_{model_name}_{feature}.pt")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs. Best val_loss: {best_val_loss:.4f} | Best val_mape: {best_val_mape:.2f}%")
                break

    return best_val_loss, best_val_mape

def run_model_comparison(elite_data, input_size, num_epochs, feature=None):
    results = []
    train_samples, val_samples = train_test_split(elite_data, test_size=0.2, random_state=42)
    train_dataset = GaitCycleDataset(train_samples)
    val_dataset = GaitCycleDataset(val_samples)

    combinations = [
        ('LSTM', 'LSTM'),
        ('GRU', 'GRU'),
        ('RNN', 'RNN'),
        ('LSTM', 'GRU'),
        ('GRU', 'LSTM'),
        ('RNN', 'GRU'),
        ('LSTM', 'RNN'),
        ('GRU', 'RNN'),
        ('RNN', 'LSTM')
    ]

    learning_rates = [1e-2, 0.005, 1e-3]
    batch_sizes = [8, 16]
    patience_values = [50]

    total_runs = len(combinations) * len(learning_rates) * len(batch_sizes) * len(patience_values)
    pbar = tqdm(total=total_runs, desc="Grid Search")

    for enc, dec in combinations:
        for lr in learning_rates:
            for bs in batch_sizes:
                for pt in patience_values:
                    model = create_model(input_size, enc, dec)
                    model_name = f"{enc}_{dec}_lr{lr}_bs{bs}_pt{pt}"
                    val_loss, val_mape = train_model_with_validation(
                        model, train_dataset, val_dataset, model_name,
                        num_epochs=num_epochs, batch_size=bs,
                        learning_rate=lr, patience=pt, feature=feature
                    )
                    results.append({
                        "encoder": enc,
                        "decoder": dec,
                        "learning_rate": lr,
                        "batch_size": bs,
                        "patience": pt,
                        "val_loss": val_loss,
                        "val_mape": val_mape
                    })
                    pbar.update(1)

    pbar.close()
    df_results = pd.DataFrame(results)
    print("\n=== Hyperparameter Search Summary ===")
    print(df_results.sort_values("val_loss"))
    df_results.to_csv(f"hyperparameter_search_results_{feature}.csv", index=False)
    return df_results

# ===== Load Trained Model (.pt) =====
def load_model(model_path, input_size, encoder_type, decoder_type):
    model = create_model(input_size, encoder_type=encoder_type, decoder_type=decoder_type)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

# ===== Evaluate User Gait (MAPE + Error Plot) =====
def evaluate_user_gait(model, user_data):
    """
    user_data: numpy array [T, F]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    user_tensor = torch.tensor(user_data, dtype=torch.float32).unsqueeze(0).to(device)  # [1, T, F]

    with torch.no_grad():
        output = model(user_tensor)
        output_np = output.squeeze(0).cpu().numpy()
        original_np = user_tensor.squeeze(0).cpu().numpy()

        print(f'input: {original_np}')
        print(f'output: {output_np}')
        print(f'input_len: {len(original_np)}')
        print(f'output_len: {len(output_np)}')

        # Compute MAPE
        mape = np.mean(np.abs((original_np - output_np) / (original_np + 1e-8))) * 100

        # Compute MSE per frame (for plotting)
        frame_errors = np.mean((original_np - output_np) ** 2, axis=1)

    print(f"📊 User MAPE: {mape:.2f}%")

    # Plot reconstruction error
    plt.plot(frame_errors)
    plt.title("Reconstruction Error per Frame (MSE)")
    plt.xlabel("Frame Index")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return mape, frame_errors

def evaluate_models_and_collect_mape(pose_df, model_dir):
    labeled_pose_data, gait_cycles, left_strikes, right_strikes = process_and_label_gait_cycles(pose_df)

    feature_mape_dict = {}

    model_files = glob.glob(os.path.join(model_dir, "best_*_lr*_bs*_pt*_*.pt"))
    print(f"🧠 Found {len(model_files)} model files")

    for model_path in model_files:
        model_filename = os.path.basename(model_path)
        parts = model_filename.split("_")

        if len(parts) < 7:
            print(f"⚠️ Skipping malformed filename: {model_filename}")
            continue

        encoder_type = parts[1]
        decoder_type = parts[2]
        feature = "_".join(parts[6:]).replace(".pt", "")

        print(f"\n📦 Loading model: {model_filename}")
        print(f"🔍 Feature: {feature} | Encoder: {encoder_type} | Decoder: {decoder_type}")

        model = load_model(model_path, input_size=1, encoder_type=encoder_type, decoder_type=decoder_type)

        if feature not in labeled_pose_data.columns:
            print(f"❌ Feature '{feature}' not found in pose_df")
            continue

        # single_feature_df = pose_df[[feature]].copy()
        result = evaluate_gait_anomaly(model, pose_df, [feature])

        if isinstance(result, tuple):
            _, _, mape_avg, _, recon, x_flat = result
            print(f"✅ {feature}: Avg MAPE = {mape_avg:.2f}%")
            feature_mape_dict[feature] = mape_avg

            # plot_reconstruction_overlay(
            #     x_flat, recon,
            #     feature_name=feature,
            #     sample_idx=0
            # )
            # plot_reconstruction_error_curve(
            #     x_flat, recon,
            #     feature_name=feature
            # )

        else:
            print(f"⚠️ Error evaluating {feature}: {result}")

    return feature_mape_dict

# --- Main Entry Point ---
if __name__ == "__main__":
    elite_video_paths = ["1.mov", "2.mov", "3.mov", "Eliud Kipchoge - The Final Kilometre of the INEOS -Edit.MOV", "Eliud Kipchoge Marathon World Record 2022-Edit.MOV",
                         "Kipchoge-Edit.MOV", "Paul Chelimo-Edit2.mp4", "Paul Chelimo-Edit1.mp4", "Noah Lyles cruises-Edit.MOV"]

    selected_features = [
        'trunk_lean', 'stride_width', 'center_of_mass_y', 'arm_swing_ratio',
        'leading_elbow_angle', 'trailing_elbow_angle',
        'leading_hip_angle', 'trailing_hip_angle',
        'leading_ankle_x', 'trailing_ankle_x',
        'leading_ankle_y', 'trailing_ankle_y',
        'leading_knee_angle', 'trailing_knee_angle'
    ]

    for feature in selected_features:
        print(f'************ Feature: {feature} ************')
        elite_data = load_gait_cycles_from_videos(elite_video_paths, [feature], num_points=100)
        run_model_comparison(elite_data, input_size=1, num_epochs=2000, feature=feature)