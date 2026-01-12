import torch
import torch.nn as nn
import cv2
import numpy as np
from typing import Dict, Optional
import logging
import os # Added for path checking

logger = logging.getLogger(__name__)

class PhysiologicalDetector(nn.Module):
    """
    Physiological detector analyzing rPPG and lip-sync consistency
    to detect deepfakes based on biological signal inconsistencies.
    """

    def __init__(self, num_classes: int = 2, sequence_length: int = 16):
        super(PhysiologicalDetector, self).__init__()

        self.num_classes = num_classes
        self.sequence_length = sequence_length

        # rPPG signal analyzer
        self.rppg_analyzer = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=5, padding=2),  # RGB channels
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        # Lip-sync analyzer with LSTM
        self.lipsync_analyzer = nn.LSTM(
            input_size=136,  # 68 landmarks * 2 coordinates
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # Classification head
        total_features = 128 + 256  # rPPG + LSTM features

        self.classifier = nn.Sequential(
            nn.Linear(total_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(128, num_classes)
        )

        # Simple face detection (would use dlib in practice)
        self.face_cascade = None
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception as e:
            logger.warning(f"Could not load face cascade: {e}")

    def extract_rppg_signal(self, video_frames: torch.Tensor) -> torch.Tensor:
        """Extract rPPG signal from facial regions."""
        batch_size, seq_len = video_frames.shape[:2]
        rppg_signals = []

        for b in range(batch_size):
            batch_rppg = []

            for t in range(seq_len):
                frame = video_frames[b, t].cpu().numpy().transpose(1, 2, 0)
                frame = (frame * 255).astype(np.uint8)

                if self.face_cascade is not None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

                    if len(faces) > 0:
                        face = faces[0]  # Use first detected face
                        x, y, w, h = face
                        face_region = frame[y:y+h, x:x+w]

                        # Extract RGB means
                        r_mean = np.mean(face_region[:, :, 0])
                        g_mean = np.mean(face_region[:, :, 1])  
                        b_mean = np.mean(face_region[:, :, 2])
                        batch_rppg.append([r_mean, g_mean, b_mean])
                    else:
                        # Fallback: use entire frame if face detection fails
                        r_mean = np.mean(frame[:, :, 0])
                        g_mean = np.mean(frame[:, :, 1])
                        b_mean = np.mean(frame[:, :, 2])
                        batch_rppg.append([r_mean, g_mean, b_mean])
                else:
                    # Fallback: use entire frame
                    r_mean = np.mean(frame[:, :, 0])
                    g_mean = np.mean(frame[:, :, 1])
                    b_mean = np.mean(frame[:, :, 2])
                    batch_rppg.append([r_mean, g_mean, b_mean])

            # Ensure consistent sequence length
            while len(batch_rppg) < seq_len:
                batch_rppg.append([128.0, 128.0, 128.0])

            rppg_signals.append(batch_rppg[:seq_len])

        return torch.FloatTensor(rppg_signals).transpose(1, 2).to(video_frames.device)

    def extract_facial_landmarks(self, video_frames: torch.Tensor) -> torch.Tensor:
        """Extract facial landmark sequences using dlib."""
        try:
            import dlib
        except ImportError:
            logger.warning("dlib not installed, using dummy landmarks")
            return self._extract_dummy_landmarks(video_frames)
            
        if not hasattr(self, 'detector'):
            self.detector = dlib.get_frontal_face_detector()
            logger.info("Initialized Dlib Face Detector")
            
        if not hasattr(self, 'predictor'):
            # Check for shape predictor file using project-relative path
            import pathlib
            project_root = pathlib.Path(__file__).parent.parent.parent.parent
            predictor_path = project_root / "models" / "shape_predictor_68_face_landmarks.dat"
            
            if not predictor_path.exists():
                 if not hasattr(self, '_warned_predictor'):
                     logger.warning(f"Predictor not found at {predictor_path}. Using dummy landmarks.")
                     self._warned_predictor = True
                 return self._extract_dummy_landmarks(video_frames)
            self.predictor = dlib.shape_predictor(str(predictor_path))

        batch_size, seq_len = video_frames.shape[:2]
        landmark_sequences = []
        
        faces_found = 0

        for b in range(batch_size):
            batch_landmarks = []

            for t in range(seq_len):
                frame = video_frames[b, t].cpu().numpy().transpose(1, 2, 0)
                frame = (frame * 255).astype(np.uint8)
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                
                faces = self.detector(gray, 0) # 0 for no upsampling (faster)
                if len(faces) > 0:
                    faces_found += 1
                    shape = self.predictor(gray, faces[0])
                    landmarks = []
                    for i in range(68):
                        landmarks.extend([shape.part(i).x, shape.part(i).y])
                    batch_landmarks.append(landmarks)
                else:
                    # Fallback: Assume whole image is face
                    h, w = gray.shape
                    rect = dlib.rectangle(0, 0, w, h)
                    shape = self.predictor(gray, rect)
                    landmarks = []
                    for i in range(68):
                        landmarks.extend([shape.part(i).x, shape.part(i).y])
                    batch_landmarks.append(landmarks)

            # Ensure consistent sequence length
            while len(batch_landmarks) < seq_len:
                batch_landmarks.append([0.0] * 136)

            landmark_sequences.append(batch_landmarks[:seq_len])
            
        if faces_found == 0 and not hasattr(self, '_warned_nofaces'):
             logger.warning(f"No faces detected in batch! Using full-frame fallback for physiological analysis.")
             self._warned_nofaces = True
        elif faces_found > 0 and not hasattr(self, '_logged_success'):
             logger.info(f"Dlib successfully detected faces in {faces_found} frames.")
             self._logged_success = True

        return torch.FloatTensor(landmark_sequences).to(video_frames.device)

    def _extract_dummy_landmarks(self, video_frames: torch.Tensor) -> torch.Tensor:
         # ... original dummy logic ...
        batch_size, seq_len = video_frames.shape[:2]
        landmark_sequences = []

        for b in range(batch_size):
            batch_landmarks = []
            for t in range(seq_len):
                landmarks = []
                for i in range(68):
                    x_offset = (i % 17) * 224 / 17
                    y_offset = (i // 17) * 224 / 4
                    landmarks.extend([x_offset, y_offset])
                batch_landmarks.append(landmarks)
            landmark_sequences.append(batch_landmarks)
            
        return torch.FloatTensor(landmark_sequences).to(video_frames.device)

    def forward(self, video_frames: torch.Tensor, audio_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through physiological detector."""
        # Extract physiological signals
        rppg_signal = self.extract_rppg_signal(video_frames)
        facial_landmarks = self.extract_facial_landmarks(video_frames)

        # Process rPPG signal
        rppg_features = self.rppg_analyzer(rppg_signal)

        # Process lip-sync features
        lstm_output, _ = self.lipsync_analyzer(facial_landmarks)
        lipsync_features = lstm_output[:, -1, :]  # Take final hidden state

        # Combine features
        combined_features = torch.cat([rppg_features, lipsync_features], dim=1)

        # Classification
        return self.classifier(combined_features)

    def get_physiological_analysis(self, video_frames: torch.Tensor) -> Dict[str, np.ndarray]:
        """Get physiological analysis for explainability."""
        with torch.no_grad():
            rppg_signal = self.extract_rppg_signal(video_frames)
            landmarks = self.extract_facial_landmarks(video_frames)

            return {
                'rppg_signal': rppg_signal.cpu().numpy(),
                'facial_landmarks': landmarks.cpu().numpy(),
                'signal_quality': torch.std(rppg_signal, dim=2).cpu().numpy()
            }

def create_physiological_detector(num_classes: int = 2, sequence_length: int = 16) -> PhysiologicalDetector:
    """Factory function to create a physiological detector model.""" 
    return PhysiologicalDetector(num_classes=num_classes, sequence_length=sequence_length)
