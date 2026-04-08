import sys
import os
import glob
import threading
import tempfile

import cv2 as cv
import numpy as np
from scipy import linalg

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget,
    QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit,
    QLabel, QSpinBox, QDoubleSpinBox, QFileDialog,
    QProgressBar, QSlider, QScrollArea, QGridLayout,
    QDialog, QSplitter,
)
from PyQt6.QtGui import QFont, QPixmap, QImage, QPainter, QPen, QBrush, QColor
from PyQt6.QtCore import QThread, QObject, pyqtSignal, Qt, QTimer, QSize

try:
    import open3d as o3d
    _O3D_AVAILABLE = True
except ImportError:
    _O3D_AVAILABLE = False


# ── Utilitaire zoom ───────────────────────────────────────────────────────────

def get_zoomed_patch(image: np.ndarray, x: int, y: int,
                     patch_size: int = 75, display_size: int = 300) -> np.ndarray:
    h, w = image.shape[:2]
    x1 = max(0, x - patch_size // 2)
    y1 = max(0, y - patch_size // 2)
    x2 = min(w, x + patch_size // 2)
    y2 = min(h, y + patch_size // 2)
    patch  = image[y1:y2, x1:x2]
    zoomed = cv.resize(patch, (display_size, display_size),
                       interpolation=cv.INTER_LINEAR)
    cx, cy = display_size // 2, display_size // 2
    cv.line(zoomed, (cx - 15, cy), (cx + 15, cy), (0, 255, 0), 1)
    cv.line(zoomed, (cx, cy - 15), (cx, cy + 15), (0, 255, 0), 1)
    return zoomed


# ── Fonctions de calibration ───────────────────────────────────────────────────

def DLT(P1, P2, point1, point2):
    A = [point1[1] * P1[2, :] - P1[1, :],
         P1[0, :] - point1[0] * P1[2, :],
         point2[1] * P2[2, :] - P2[1, :],
         P2[0, :] - point2[0] * P2[2, :]]
    A = np.array(A).reshape((4, 4))
    B = A.transpose() @ A
    U, s, Vh = linalg.svd(B, full_matrices=False)
    return Vh[3, 0:3] / Vh[3, 3]


def extract_frames(video_path, output_folder, frame_indices):
    cap = cv.VideoCapture(video_path)
    os.makedirs(output_folder, exist_ok=True)
    saved = []
    for idx in frame_indices:
        cap.set(cv.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            path = os.path.join(output_folder, f"{idx:06d}.png")
            cv.imwrite(path, frame)
            saved.append(path)
    cap.release()
    return saved


def calibrate_camera(image_paths, rows, columns, square_size_mm, log_fn):
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp *= square_size_mm
    objpoints, imgpoints = [], []
    detected_info = []          # per-frame info for verification UI
    images = [cv.imread(p, 1) for p in image_paths]
    images = [im for im in images if im is not None]
    if not images:
        raise RuntimeError("No images loaded.")
    h, w = images[0].shape[:2]
    detected = 0
    for frame, path in zip(images, image_paths):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)
        if ret:
            corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp.copy())
            imgpoints.append(corners)
            annotated = frame.copy()
            cv.drawChessboardCorners(annotated, (rows, columns), corners, True)
            detected_info.append({
                'path':            path,
                'frame':           frame.copy(),
                'frame_annotated': annotated,
                'corners':         corners.copy(),   # shape (N,1,2) float32
            })
            detected += 1
    log_fn(f"  Checkerboard detected: {detected}/{len(images)} frames")
    if not objpoints:
        raise RuntimeError("No checkerboard detected. Try swapping rows/columns.")
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, (w, h), None, None)
    total_error = 0
    for i in range(len(objpoints)):
        pts2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        total_error += cv.norm(imgpoints[i], pts2, cv.NORM_L2) / len(pts2)
    pixel_rmse = total_error / len(objpoints)
    log_fn(f"  Reprojection error: {pixel_rmse:.4f} pixels")
    return mtx, dist, pixel_rmse, objpoints, imgpoints, detected_info, (w, h)


def stereo_calibrate(mtx1, dist1, mtx2, dist2,
                     left_paths, right_paths,
                     rows, columns, square_size_mm, log_fn):
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp *= square_size_mm
    objpoints, imgpoints_left, imgpoints_right = [], [], []
    valid = 0
    for p1, p2 in zip(left_paths, right_paths):
        f1 = cv.imread(p1, 1)
        f2 = cv.imread(p2, 1)
        if f1 is None or f2 is None:
            continue
        g1 = cv.cvtColor(f1, cv.COLOR_BGR2GRAY)
        g2 = cv.cvtColor(f2, cv.COLOR_BGR2GRAY)
        r1, c1 = cv.findChessboardCorners(g1, (rows, columns), None)
        r2, c2 = cv.findChessboardCorners(g2, (rows, columns), None)
        if r1 and r2:
            c1 = cv.cornerSubPix(g1, c1, (11, 11), (-1, -1), criteria)
            c2 = cv.cornerSubPix(g2, c2, (11, 11), (-1, -1), criteria)
            objpoints.append(objp.copy())
            imgpoints_left.append(c1)
            imgpoints_right.append(c2)
            valid += 1
    log_fn(f"  Valid stereo pairs: {valid}/{len(left_paths)}")
    if not objpoints:
        raise RuntimeError("No valid stereo pairs found.")
    h, w = cv.imread(left_paths[0]).shape[:2]
    flags = cv.CALIB_FIX_INTRINSIC
    ret, _, _, _, _, R, T, E, F = cv.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        mtx1, dist1, mtx2, dist2, (w, h),
        criteria=criteria, flags=flags)
    log_fn(f"  Stereo RMSE: {ret:.4f} pixels")
    return R, T, F, ret


# ── Worker calibration ─────────────────────────────────────────────────────────

class CalibrationWorker(QObject):
    log          = pyqtSignal(str)
    progress     = pyqtSignal(int)
    finished     = pyqtSignal()
    error        = pyqtSignal(str)
    verify_ready = pyqtSignal(object)   # emits dict with verify data

    left_video:     str   = ""
    right_video:    str   = ""
    rows:           int   = 4
    columns:        int   = 8
    square_size_mm: float = 63.42

    def run(self):
        try:
            os.makedirs('camera_parameters', exist_ok=True)
            tmp_base  = os.path.join(tempfile.gettempdir(), 'stereo_calib')
            tmp_left  = os.path.join(tmp_base, 'left')
            tmp_right = os.path.join(tmp_base, 'right')

            # ── Étape 1 : extraction des frames ───────────────────────────
            self.log.emit("── Step 1: Extracting frames ──────────────────────")
            cap = cv.VideoCapture(self.left_video)
            total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
            cap.release()
            N = max(1, total // 80)
            frame_indices = list(range(0, total, N))[:80]
            self.log.emit(
                f"  Total frames: {total}  |  extracting {len(frame_indices)} "
                f"frames (every {N})")
            self.progress.emit(5)

            left_paths = extract_frames(self.left_video, tmp_left, frame_indices)
            self.log.emit(f"  Left  : {len(left_paths)} frames saved → {tmp_left}")
            self.progress.emit(20)

            right_paths = extract_frames(self.right_video, tmp_right, frame_indices)
            self.log.emit(f"  Right : {len(right_paths)} frames saved → {tmp_right}")
            self.progress.emit(35)

            # ── Étape 2 : détection de l'orientation du damier ────────────
            rows, columns = self.rows, self.columns
            self.log.emit("")
            self.log.emit("── Step 2: Checkerboard orientation ───────────────")
            img  = cv.imread(left_paths[0], 1)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            ret, _ = cv.findChessboardCorners(gray, (rows, columns), None)
            if ret:
                self.log.emit(f"  OK — using ({rows}, {columns})")
            else:
                ret2, _ = cv.findChessboardCorners(gray, (columns, rows), None)
                if ret2:
                    rows, columns = columns, rows
                    self.log.emit(f"  Swapped → using ({rows}, {columns})")
                else:
                    self.log.emit(
                        f"  WARNING: checkerboard not detected on first frame "
                        f"with ({rows},{columns}) or ({columns},{rows})")
            self.progress.emit(40)

            # ── Étape 3 : calibration intrinsèque ─────────────────────────
            self.log.emit("")
            self.log.emit("── Step 3: Intrinsic calibration — Left ───────────")
            mtx1, dist1, rmse1, objpts_l, imgpts_l, info_l, sz_l = calibrate_camera(
                left_paths, rows, columns, self.square_size_mm, self.log.emit)
            self.log.emit(f"  RMSE left  : {rmse1:.4f} px")
            self.progress.emit(60)

            self.log.emit("")
            self.log.emit("── Step 3: Intrinsic calibration — Right ──────────")
            mtx2, dist2, rmse2, objpts_r, imgpts_r, info_r, sz_r = calibrate_camera(
                right_paths, rows, columns, self.square_size_mm, self.log.emit)
            self.log.emit(f"  RMSE right : {rmse2:.4f} px")
            self.progress.emit(75)

            # ── Étape 4 : calibration stéréo ──────────────────────────────
            self.log.emit("")
            self.log.emit("── Step 4: Stereo calibration ─────────────────────")
            R, T, F, rmse3 = stereo_calibrate(
                mtx1, dist1, mtx2, dist2,
                left_paths, right_paths,
                rows, columns, self.square_size_mm, self.log.emit)
            self.progress.emit(90)

            # ── Étape 5 : sauvegarde ───────────────────────────────────────
            self.log.emit("")
            self.log.emit("── Step 5: Saving parameters ──────────────────────")
            np.save('camera_parameters/mtx1.npy',  mtx1)
            np.save('camera_parameters/dist1.npy', dist1)
            np.save('camera_parameters/mtx2.npy',  mtx2)
            np.save('camera_parameters/dist2.npy', dist2)
            np.save('camera_parameters/R.npy',     R)
            np.save('camera_parameters/T.npy',     T)
            np.save('camera_parameters/F.npy',     F)
            with open('camera_parameters/videos.txt', 'w') as f:
                f.write(self.left_video  + '\n')
                f.write(self.right_video + '\n')
            self.log.emit("  mtx1.npy  dist1.npy  → caméra gauche")
            self.log.emit("  mtx2.npy  dist2.npy  → caméra droite")
            self.log.emit("  R.npy  T.npy  F.npy  → stéréo (rotation, translation, fondamentale)")
            self.log.emit("  videos.txt            → chemins des vidéos")
            self.log.emit("✓ Paramètres enregistrés")

            # ── Sauvegarde des images de vérification ──────────────────────
            self.log.emit("")
            self.log.emit("── Step 6: Saving verify images ───────────────────")
            verify_dir = 'camera_parameters/verify'
            os.makedirs(verify_dir, exist_ok=True)
            for i, item in enumerate(info_l):
                item['label'] = f"Left {i+1:03d}"
                cv.imwrite(os.path.join(verify_dir, f"left_{i:04d}.png"),
                           item['frame_annotated'])
            for i, item in enumerate(info_r):
                item['label'] = f"Right {i+1:03d}"
                cv.imwrite(os.path.join(verify_dir, f"right_{i:04d}.png"),
                           item['frame_annotated'])

            # Sauvegarde des coins et objpoints pour restauration au démarrage
            np.save(os.path.join(verify_dir, 'corners_left.npy'),
                    np.array([it['corners'] for it in info_l]))
            np.save(os.path.join(verify_dir, 'corners_right.npy'),
                    np.array([it['corners'] for it in info_r]))
            np.save(os.path.join(verify_dir, 'objpoints_left.npy'),
                    np.array(objpts_l))
            np.save(os.path.join(verify_dir, 'objpoints_right.npy'),
                    np.array(objpts_r))
            np.save(os.path.join(verify_dir, 'image_sizes.npy'),
                    np.array([list(sz_l), list(sz_r)]))

            self.log.emit(
                f"  verify/left_XXXX.png   × {len(info_l)} frames gauche")
            self.log.emit(
                f"  verify/right_XXXX.png  × {len(info_r)} frames droite")
            self.log.emit("  verify/corners_left/right.npy  → coins détectés")
            self.log.emit("  verify/objpoints_left/right.npy → points 3D damier")
            self.log.emit("  verify/image_sizes.npy          → résolutions")
            self.log.emit("✓ Données de vérification enregistrées")

            self.progress.emit(100)
            self.log.emit("")
            self.log.emit("══════════════════════════════════════════════════")
            self.log.emit("✓ Calibration terminée — tous les paramètres sont enregistrés.")
            self.log.emit("  Redémarrez l'app : la calibration sera rechargée automatiquement.")
            self.log.emit("══════════════════════════════════════════════════")

            # Émettre les données pour la grille de vérification
            self.verify_ready.emit({
                'left':            info_l,
                'right':           info_r,
                'objpoints_left':  objpts_l,
                'objpoints_right': objpts_r,
                'image_size_left':  sz_l,
                'image_size_right': sz_r,
            })

        except Exception as exc:
            self.error.emit(f"[ERROR]  {exc}")
        finally:
            self.finished.emit()


# ── Onglet Calibration ────────────────────────────────────────────────────────

class CalibrationTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 12, 12, 12)

        self._left_video  = ""
        self._right_video = ""

        # ── Sélecteurs vidéo ─────────────────────────────────────────────────
        for side in ('Left', 'Right'):
            row = QHBoxLayout()
            btn = QPushButton(f"Select {side} Video")
            btn.setFixedWidth(160)
            btn.clicked.connect(lambda _, s=side.lower(): self._pick_video(s))
            lbl = QLabel("No file selected")
            lbl.setWordWrap(True)
            row.addWidget(btn)
            row.addWidget(lbl, stretch=1)
            layout.addLayout(row)
            setattr(self, f'_lbl_{side.lower()}', lbl)

        # ── Paramètres ───────────────────────────────────────────────────────
        params = QHBoxLayout()
        params.setSpacing(10)

        params.addWidget(QLabel("Rows:"))
        self.spin_rows = QSpinBox()
        self.spin_rows.setRange(2, 30)
        self.spin_rows.setValue(4)
        params.addWidget(self.spin_rows)

        params.addWidget(QLabel("Columns:"))
        self.spin_cols = QSpinBox()
        self.spin_cols.setRange(2, 30)
        self.spin_cols.setValue(8)
        params.addWidget(self.spin_cols)

        params.addWidget(QLabel("Square size (mm):"))
        self.spin_square = QDoubleSpinBox()
        self.spin_square.setRange(0.1, 1000.0)
        self.spin_square.setValue(63.42)
        self.spin_square.setDecimals(2)
        self.spin_square.setFixedWidth(90)
        params.addWidget(self.spin_square)

        params.addStretch()
        self.btn_run = QPushButton("Run Calibration")
        self.btn_run.setFixedHeight(34)
        self.btn_run.clicked.connect(self._run)
        params.addWidget(self.btn_run)
        layout.addLayout(params)

        # ── Progress bar ─────────────────────────────────────────────────────
        self.progress = QProgressBar()
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        # ── Splitter : logs (haut) / grille de vérification (bas) ────────────
        self._splitter = QSplitter(Qt.Orientation.Vertical)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setFont(QFont("Courier New", 9))
        self._splitter.addWidget(self.log_area)

        # Container pour la VerifyGrid — on y injecte le contenu après calibration
        self._verify_container = QWidget()
        self._verify_container_layout = QVBoxLayout(self._verify_container)
        self._verify_container_layout.setContentsMargins(0, 0, 0, 0)
        self._verify_placeholder = QLabel(
            "La grille de vérification apparaîtra ici après la calibration.")
        self._verify_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._verify_placeholder.setStyleSheet("color:#888; font-size:12px;")
        self._verify_container_layout.addWidget(self._verify_placeholder)

        self._splitter.addWidget(self._verify_container)
        self._splitter.setSizes([300, 0])

        layout.addWidget(self._splitter, stretch=1)

        # Restauration automatique si une calibration précédente existe
        self._try_restore()

    # ── Restauration au démarrage ─────────────────────────────────────────────

    def _try_restore(self):
        """Recharge la calibration précédente si les fichiers existent."""
        required = [
            'camera_parameters/mtx1.npy',  'camera_parameters/dist1.npy',
            'camera_parameters/mtx2.npy',  'camera_parameters/dist2.npy',
            'camera_parameters/R.npy',     'camera_parameters/T.npy',
        ]
        if any(not os.path.exists(f) for f in required):
            return

        self.log_area.append("✓ Previous calibration found in camera_parameters/")

        # Restaurer les chemins vidéo
        videos_txt = 'camera_parameters/videos.txt'
        if os.path.exists(videos_txt):
            try:
                with open(videos_txt) as f:
                    lines = [l.strip() for l in f.readlines()]
                if len(lines) >= 2 and lines[0] and lines[1]:
                    self._left_video  = lines[0]
                    self._right_video = lines[1]
                    self._lbl_left.setText(lines[0])
                    self._lbl_right.setText(lines[1])
                    self.log_area.append(f"  Left  : {lines[0]}")
                    self.log_area.append(f"  Right : {lines[1]}")
            except Exception:
                pass

        # Restaurer la grille de vérification
        vdir = 'camera_parameters/verify'
        cl   = os.path.join(vdir, 'corners_left.npy')
        cr   = os.path.join(vdir, 'corners_right.npy')
        ol   = os.path.join(vdir, 'objpoints_left.npy')
        orr  = os.path.join(vdir, 'objpoints_right.npy')
        sz   = os.path.join(vdir, 'image_sizes.npy')

        if not all(os.path.exists(p) for p in (cl, cr, ol, orr, sz)):
            self.log_area.append("  (No verify grid — run calibration to generate it)")
            return

        try:
            corners_l  = np.load(cl)    # (N, K, 1, 2)
            corners_r  = np.load(cr)
            objpts_l   = list(np.load(ol))
            objpts_r   = list(np.load(orr))
            sizes      = np.load(sz).astype(int)
            sz_l       = tuple(sizes[0])
            sz_r       = tuple(sizes[1])

            def load_side(corners_arr, prefix, side_name):
                items = []
                for i, c in enumerate(corners_arr):
                    img_path = os.path.join(vdir, f"{prefix}_{i:04d}.png")
                    if not os.path.exists(img_path):
                        continue
                    frame_ann = cv.imread(img_path, 1)
                    if frame_ann is None:
                        continue
                    items.append({
                        'path':            img_path,
                        'frame':           frame_ann,
                        'frame_annotated': frame_ann,
                        'corners':         c.astype(np.float32),
                        'label':           f"{side_name} {i+1:03d}",
                    })
                return items

            info_l = load_side(corners_l, 'left',  'Left')
            info_r = load_side(corners_r, 'right', 'Right')

            self.log_area.append(
                f"  Verify grid: {len(info_l)} left + {len(info_r)} right frames")

            if info_l or info_r:
                self._show_verify_grid({
                    'left':             info_l,
                    'right':            info_r,
                    'objpoints_left':   objpts_l,
                    'objpoints_right':  objpts_r,
                    'image_size_left':  sz_l,
                    'image_size_right': sz_r,
                })
        except Exception as exc:
            import traceback
            self.log_area.append(
                f"  [RESTORE WARNING] Could not load verify grid: {exc}\n"
                f"{traceback.format_exc()}")

    def _pick_video(self, side: str):
        path, _ = QFileDialog.getOpenFileName(
            self, f"Select {side.capitalize()} Video", "",
            "Videos (*.mp4 *.avi *.mov *.mkv *.MOV);;All Files (*)")
        if path:
            setattr(self, f'_{side}_video', path)
            getattr(self, f'_lbl_{side}').setText(path)

    def _run(self):
        if not self._left_video or not self._right_video:
            self.log_area.append("[ERROR] Select both videos first.")
            return
        self.btn_run.setEnabled(False)
        self.progress.setValue(0)
        self.log_area.clear()
        self.log_area.append("Starting calibration…\n")

        self._thread = QThread()
        self._worker = CalibrationWorker()
        self._worker.left_video     = self._left_video
        self._worker.right_video    = self._right_video
        self._worker.rows           = self.spin_rows.value()
        self._worker.columns        = self.spin_cols.value()
        self._worker.square_size_mm = self.spin_square.value()
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self.log_area.append)
        self._worker.progress.connect(self.progress.setValue)
        self._worker.error.connect(self.log_area.append)
        self._worker.verify_ready.connect(self._show_verify_grid)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(lambda: self.btn_run.setEnabled(True))

        self._thread.start()

    def _show_verify_grid(self, verify_data: dict):
        try:
            # Vider le container (supprime le placeholder ou une ancienne grille)
            while self._verify_container_layout.count():
                item = self._verify_container_layout.takeAt(0)
                w = item.widget()
                if w:
                    w.deleteLater()

            grid = VerifyGrid(verify_data, self.log_area.append,
                              parent=self._verify_container)
            self._verify_container_layout.addWidget(grid)

            # Ouvrir le panneau bas du splitter
            total = self._splitter.height()
            top   = max(200, total - 450) if total > 300 else 300
            self._splitter.setSizes([top, 450])
            self._splitter.update()
        except Exception as exc:
            import traceback
            self.log_area.append(f"[VERIFY ERROR] {exc}\n{traceback.format_exc()}")


# ── Signaux thread → UI ────────────────────────────────────────────────────────

class _PCSignals(QObject):
    status      = pyqtSignal(str)
    finished    = pyqtSignal()
    error       = pyqtSignal(str)
    image_ready = pyqtSignal(object)   # émet un np.ndarray BGR pour affichage Qt


# ── Fenêtre de visualisation disparité ────────────────────────────────────────

class DisparityWindow(QWidget):
    """Fenêtre indépendante affichant le composite disparité/rectification."""

    def __init__(self, bgr_image: np.ndarray, parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("AquaMeasure — Disparity & Rectification Check")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Barre d'info en haut
        info = QLabel(
            "Lignes épipolaires vertes alignées horizontalement → rectification OK   |   "
            "Gradient cohérent sur la carte → calibration stéréo OK   |   "
            "Noir = pas de profondeur")
        info.setStyleSheet("color:#aaa; font-size:10px;")
        info.setWordWrap(True)
        layout.addWidget(info)

        scroll = QScrollArea()
        scroll.setWidgetResizable(False)

        lbl = QLabel()
        lbl.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        pix = _cv_to_pixmap(bgr_image)
        lbl.setPixmap(pix)
        lbl.setFixedSize(pix.size())

        scroll.setWidget(lbl)
        layout.addWidget(scroll)

        # Taille fenêtre : max 90 % de l'écran
        screen = QApplication.primaryScreen().availableGeometry()
        self.resize(min(bgr_image.shape[1] + 24, int(screen.width()  * 0.92)),
                    min(bgr_image.shape[0] + 80,  int(screen.height() * 0.88)))

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)


# ── ZoomPanel ─────────────────────────────────────────────────────────────────

class ZoomPanel(QLabel):
    """Affiche un patch 4× centré sur la position souris."""

    SIZE = 300

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(self.SIZE, self.SIZE)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background:#1e1e1e; border:1px solid #555;")
        self.setText("—")

    def update_zoom(self, image: np.ndarray, x: int, y: int):
        zoomed = get_zoomed_patch(image, x, y, display_size=self.SIZE)
        self.setPixmap(_cv_to_pixmap(zoomed))


# ── VerifyThumbnail ────────────────────────────────────────────────────────────

class VerifyThumbnail(QLabel):
    """Miniature d'une frame de vérification ; hover → zoom, clic → éditeur."""

    hover_at  = pyqtSignal(object, int, int)   # (image, x_orig, y_orig)
    open_edit = pyqtSignal(object)             # verify_item dict

    THUMB = 160

    def __init__(self, item: dict, parent=None):
        super().__init__(parent)
        self._item  = item
        self._image = item['frame_annotated']
        self.setFixedSize(self.THUMB, self.THUMB)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("border:2px solid #555; background:#1e1e1e;")
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        thumb = cv.resize(self._image, (self.THUMB, self.THUMB))
        self.setPixmap(_cv_to_pixmap(thumb))

    def _orig(self, pos):
        h, w = self._image.shape[:2]
        x = int(max(0, min(pos.x() * w / self.THUMB, w - 1)))
        y = int(max(0, min(pos.y() * h / self.THUMB, h - 1)))
        return x, y

    def mouseMoveEvent(self, event):
        x, y = self._orig(event.position())
        self.hover_at.emit(self._image, x, y)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.open_edit.emit(self._item)


# ── CornerCanvas ───────────────────────────────────────────────────────────────

class CornerCanvas(QWidget):
    """Widget de dessin interactif pour ajuster les coins du damier."""

    HANDLE_R   = 8
    SNAP_DIST  = 20

    hover_at = pyqtSignal(object, int, int)

    def __init__(self, frame: np.ndarray, corners: np.ndarray, parent=None):
        super().__init__(parent)
        self._frame    = frame
        self._corners  = corners.reshape(-1, 2).astype(float)
        self._orig_corners = self._corners.copy()
        self._selected = None
        self.setMouseTracking(True)
        self.setMinimumSize(600, 400)
        # Pre-build base QPixmap from frame
        rgb  = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        self._orig_w = w
        self._orig_h = h
        self._base_pix = QPixmap.fromImage(
            QImage(rgb.tobytes(), w, h, ch * w, QImage.Format.Format_RGB888))

    def _transform(self):
        scale = min(self.width() / self._orig_w, self.height() / self._orig_h)
        ox = (self.width()  - self._orig_w * scale) / 2
        oy = (self.height() - self._orig_h * scale) / 2
        return scale, ox, oy

    def _to_disp(self, x, y):
        s, ox, oy = self._transform()
        return ox + x * s, oy + y * s

    def _to_orig(self, x, y):
        s, ox, oy = self._transform()
        return (x - ox) / s, (y - oy) / s

    def paintEvent(self, event):
        painter = QPainter(self)
        s, ox, oy = self._transform()
        dw = int(self._orig_w * s)
        dh = int(self._orig_h * s)
        painter.drawPixmap(
            int(ox), int(oy),
            self._base_pix.scaled(QSize(dw, dh),
                                  Qt.AspectRatioMode.IgnoreAspectRatio,
                                  Qt.TransformationMode.SmoothTransformation))
        n = len(self._corners)
        for i, (cx, cy) in enumerate(self._corners):
            dx, dy = self._to_disp(cx, cy)
            r = self.HANDLE_R + (3 if i == self._selected else 0)
            if i == 0:
                col = QColor(220, 30, 30)
            elif i == n - 1:
                col = QColor(40, 80, 255)
            else:
                col = QColor(255, 210, 0)
            painter.setBrush(QBrush(col))
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.drawEllipse(int(dx - r), int(dy - r), r * 2, r * 2)
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.drawText(int(dx + r + 3), int(dy + 5), str(i))
        painter.end()

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        px, py = event.position().x(), event.position().y()
        self._selected = None
        best = self.HANDLE_R + 6
        for i, (cx, cy) in enumerate(self._corners):
            dx, dy = self._to_disp(cx, cy)
            d = ((dx - px) ** 2 + (dy - py) ** 2) ** 0.5
            if d < best:
                best = d
                self._selected = i
        self.update()

    def mouseMoveEvent(self, event):
        px, py = event.position().x(), event.position().y()
        ox, oy = self._to_orig(px, py)
        if self._selected is not None:
            self._corners[self._selected] = [ox, oy]
            self.update()
        else:
            ix, iy = int(max(0, min(ox, self._orig_w - 1))), \
                     int(max(0, min(oy, self._orig_h - 1)))
            self.hover_at.emit(self._frame, ix, iy)

    def mouseReleaseEvent(self, event):
        if self._selected is not None:
            x, y = self._corners[self._selected]
            s, _, _ = self._transform()
            thresh = self.SNAP_DIST / s
            best, best_d = None, thresh
            for cx, cy in self._orig_corners:
                d = ((cx - x) ** 2 + (cy - y) ** 2) ** 0.5
                if d < best_d:
                    best_d = d
                    best = (cx, cy)
            if best:
                self._corners[self._selected] = list(best)
            self._selected = None
            self.update()

    def get_corners(self) -> np.ndarray:
        return self._corners.reshape(-1, 1, 2).astype(np.float32)

    def reset(self):
        self._corners = self._orig_corners.copy()
        self._selected = None
        self.update()


# ── CornerEditorDialog ─────────────────────────────────────────────────────────

class CornerEditorDialog(QDialog):
    """Vue plein écran avec handles déplaçables et panneau zoom."""

    def __init__(self, item: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Corner Editor — {item['label']}")
        self.resize(1280, 780)

        layout = QVBoxLayout(self)

        ctrl = QHBoxLayout()
        btn_save  = QPushButton("Save adjustments")
        btn_reset = QPushButton("Reset corners")
        btn_save.clicked.connect(self.accept)
        btn_reset.clicked.connect(lambda: self.canvas.reset())
        ctrl.addWidget(btn_save)
        ctrl.addWidget(btn_reset)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        main_row = QHBoxLayout()
        self.canvas = CornerCanvas(item['frame'], item['corners'])
        main_row.addWidget(self.canvas, stretch=1)

        zoom_col = QVBoxLayout()
        zoom_col.addWidget(QLabel("Zoom (4x)"))
        self.zoom = ZoomPanel()
        zoom_col.addWidget(self.zoom)
        zoom_col.addStretch()
        main_row.addLayout(zoom_col)
        layout.addLayout(main_row)

        self.canvas.hover_at.connect(self.zoom.update_zoom)

    def get_adjusted_corners(self) -> np.ndarray:
        return self.canvas.get_corners()


# ── VerifyGrid ─────────────────────────────────────────────────────────────────

class VerifyGrid(QWidget):
    """Grille scrollable de thumbnails + panneau zoom + recalibration."""

    COLS = 5

    def __init__(self, verify_data: dict, log_fn, parent=None):
        super().__init__(parent)
        self._data   = verify_data
        self._log_fn = log_fn
        # Copie locale des imgpoints pour permettre la modification
        self._imgpts_l = [item['corners'].copy() for item in verify_data['left']]
        self._imgpts_r = [item['corners'].copy() for item in verify_data['right']]

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # En-tête
        hdr = QHBoxLayout()
        n_l = len(verify_data['left'])
        n_r = len(verify_data['right'])
        hdr.addWidget(QLabel(
            f"<b>Vérification :</b>  {n_l} frames gauche détectées  |  "
            f"{n_r} frames droite détectées  — cliquez sur une image pour ajuster les coins"))
        hdr.addStretch()
        self.btn_recalib = QPushButton("Recalibrate with adjusted corners")
        self.btn_recalib.clicked.connect(self._recalibrate)
        hdr.addWidget(self.btn_recalib)
        layout.addLayout(hdr)

        # Grille + zoom
        row = QHBoxLayout()
        row.setSpacing(8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        grid_w = QWidget()
        grid   = QGridLayout(grid_w)
        grid.setSpacing(6)

        self.zoom_panel = ZoomPanel()

        all_items = [('L', i, it) for i, it in enumerate(verify_data['left'])] + \
                    [('R', i, it) for i, it in enumerate(verify_data['right'])]

        for pos, (side, idx, item) in enumerate(all_items):
            thumb = VerifyThumbnail(item)
            thumb.hover_at.connect(self.zoom_panel.update_zoom)
            thumb.open_edit.connect(
                lambda it, s=side, i=idx: self._open_editor(s, i, it))
            cell = QWidget()
            cl   = QVBoxLayout(cell)
            cl.setContentsMargins(2, 2, 2, 2)
            cl.setSpacing(2)
            lbl = QLabel(item['label'])
            lbl.setFont(QFont("Segoe UI", 8))
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            cl.addWidget(thumb)
            cl.addWidget(lbl)
            grid.addWidget(cell, pos // self.COLS, pos % self.COLS)

        scroll.setWidget(grid_w)
        row.addWidget(scroll, stretch=1)

        zoom_col = QVBoxLayout()
        zoom_col.addWidget(QLabel("<b>Zoom (4x)</b>"))
        zoom_col.addWidget(self.zoom_panel)
        zoom_col.addStretch()
        row.addLayout(zoom_col)
        layout.addLayout(row)

    def _open_editor(self, side: str, idx: int, item: dict):
        dlg = CornerEditorDialog(item, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            adjusted = dlg.get_adjusted_corners()
            item['corners'] = adjusted
            if side == 'L':
                self._imgpts_l[idx] = adjusted
            else:
                self._imgpts_r[idx] = adjusted
            self._log_fn(f"  Corners adjusted for {item['label']}")

    def _recalibrate(self):
        """Relance calibrateCamera avec les imgpoints courants (ajustés ou non)."""
        d = self._data
        try:
            self._log_fn("── Recalibrating with adjusted corners ────────────")
            # Gauche
            ret1, mtx1, dist1, rv1, tv1 = cv.calibrateCamera(
                d['objpoints_left'],
                [p.reshape(-1, 1, 2) for p in self._imgpts_l],
                d['image_size_left'], None, None)
            err1 = sum(
                cv.norm(self._imgpts_l[i].reshape(-1,1,2),
                        cv.projectPoints(d['objpoints_left'][i], rv1[i], tv1[i], mtx1, dist1)[0],
                        cv.NORM_L2) / len(self._imgpts_l[i])
                for i in range(len(d['objpoints_left']))
            ) / len(d['objpoints_left'])
            self._log_fn(f"  Left  RMSE: {err1:.4f} px")
            # Droite
            ret2, mtx2, dist2, rv2, tv2 = cv.calibrateCamera(
                d['objpoints_right'],
                [p.reshape(-1, 1, 2) for p in self._imgpts_r],
                d['image_size_right'], None, None)
            err2 = sum(
                cv.norm(self._imgpts_r[i].reshape(-1,1,2),
                        cv.projectPoints(d['objpoints_right'][i], rv2[i], tv2[i], mtx2, dist2)[0],
                        cv.NORM_L2) / len(self._imgpts_r[i])
                for i in range(len(d['objpoints_right']))
            ) / len(d['objpoints_right'])
            self._log_fn(f"  Right RMSE: {err2:.4f} px")
            # Sauvegarde
            np.save('camera_parameters/mtx1.npy',  mtx1)
            np.save('camera_parameters/dist1.npy', dist1)
            np.save('camera_parameters/mtx2.npy',  mtx2)
            np.save('camera_parameters/dist2.npy', dist2)
            self._log_fn("  Updated camera_parameters/mtx*.npy and dist*.npy")
        except Exception as exc:
            self._log_fn(f"[ERROR] Recalibration: {exc}")


# ── Widget image cliquable ─────────────────────────────────────────────────────

class ScaledImageLabel(QLabel):
    """Affiche une image redimensionnée avec ratio conservé. Émet clicked_at
    en coordonnées de l'image originale."""

    clicked_at = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap_orig: QPixmap | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(1, 1)
        self.setStyleSheet("background: #1e1e1e;")

    def set_pixmap(self, pixmap: QPixmap):
        self._pixmap_orig = pixmap
        self._refresh()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self):
        if self._pixmap_orig is None:
            return
        scaled = self._pixmap_orig.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        super().setPixmap(scaled)

    def mousePressEvent(self, event):
        if self._pixmap_orig is None:
            return
        orig_w = self._pixmap_orig.width()
        orig_h = self._pixmap_orig.height()
        scaled = self._pixmap_orig.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        sw, sh = scaled.width(), scaled.height()
        ox = (self.width()  - sw) / 2
        oy = (self.height() - sh) / 2
        cx = event.position().x() - ox
        cy = event.position().y() - oy
        if 0 <= cx < sw and 0 <= cy < sh:
            self.clicked_at.emit(cx * orig_w / sw, cy * orig_h / sh)


# ── Widget de mesure stéréo (handles + zoom overlay) ──────────────────────────

class MeasureImageWidget(QWidget):
    """Affiche une frame vidéo avec handles A/B draggables et zoom 4× overlay."""

    HANDLE_R = 8
    HIT_DIST = 15     # px display
    ZOOM_SZ  = 280    # px du panneau zoom

    point_placed    = pyqtSignal(str, float, float)  # (side, x_orig, y_orig)
    handle_released = pyqtSignal()

    def __init__(self, side: str, parent=None):
        super().__init__(parent)
        self._side      = side
        self._frame:    np.ndarray | None = None
        self._handles:  list = [None, None]   # [(x_orig, y_orig) | None]
        self._drag_idx: int | None = None
        self._hover:    tuple | None = None   # (x_disp, y_disp)
        self.setMouseTracking(True)
        self.setMinimumSize(200, 150)
        self.setStyleSheet("background:#1e1e1e;")

    # ── API ───────────────────────────────────────────────────────────────────

    def set_frame(self, frame: np.ndarray):
        self._frame = frame
        self.update()

    def set_handle(self, idx: int, x_orig: float, y_orig: float):
        self._handles[idx] = (x_orig, y_orig)
        self.update()

    def get_handles(self) -> list:
        return list(self._handles)

    def clear_handles(self):
        self._handles   = [None, None]
        self._drag_idx  = None
        self._hover     = None
        self.update()

    # ── Coordonnées ──────────────────────────────────────────────────────────

    def _transform(self):
        if self._frame is None:
            return 1.0, 0.0, 0.0
        h, w = self._frame.shape[:2]
        s  = min(self.width() / w, self.height() / h)
        ox = (self.width()  - w * s) / 2
        oy = (self.height() - h * s) / 2
        return s, ox, oy

    def _to_disp(self, xo, yo):
        s, ox, oy = self._transform()
        return ox + xo * s, oy + yo * s

    def _to_orig(self, xd, yd):
        s, ox, oy = self._transform()
        if self._frame is None:
            return 0.0, 0.0
        h, w = self._frame.shape[:2]
        return (max(0.0, min((xd - ox) / s, float(w - 1))),
                max(0.0, min((yd - oy) / s, float(h - 1))))

    def _find_handle(self, xd, yd) -> int | None:
        for i, h in enumerate(self._handles):
            if h is None:
                continue
            dx, dy = self._to_disp(h[0], h[1])
            if ((dx - xd) ** 2 + (dy - yd) ** 2) ** 0.5 < self.HIT_DIST + 2:
                return i
        return None

    # ── Dessin ────────────────────────────────────────────────────────────────

    def paintEvent(self, event):
        if self._frame is None:
            return
        painter = QPainter(self)
        s, ox, oy = self._transform()
        oh, ow = self._frame.shape[:2]
        dw, dh = int(ow * s), int(oh * s)

        # Frame
        rgb  = cv.cvtColor(self._frame, cv.COLOR_BGR2RGB)
        qimg = QImage(rgb.tobytes(), ow, oh, 3 * ow, QImage.Format.Format_RGB888)
        pix  = QPixmap.fromImage(qimg).scaled(
            QSize(dw, dh),
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        painter.drawPixmap(int(ox), int(oy), pix)

        # Handles + ligne
        col = QColor(220, 30, 30) if self._side == 'left' else QColor(40, 80, 255)
        for i, handle in enumerate(self._handles):
            if handle is None:
                continue
            dx, dy = self._to_disp(handle[0], handle[1])
            r = self.HANDLE_R + (4 if i == self._drag_idx else 0)
            painter.setBrush(QBrush(col))
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.drawEllipse(int(dx - r), int(dy - r), r * 2, r * 2)
            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.drawText(int(dx + r + 3), int(dy + 5), "AB"[i])

        if all(h is not None for h in self._handles):
            da = self._to_disp(*self._handles[0])
            db = self._to_disp(*self._handles[1])
            painter.setPen(QPen(col, 2))
            painter.drawLine(int(da[0]), int(da[1]), int(db[0]), int(db[1]))

        # Zoom overlay (coin haut-droit)
        if self._hover is not None:
            hxd, hyd = self._hover
            hxo, hyo = self._to_orig(hxd, hyd)

            # Dessiner les handles sur la frame avant zoom
            zoom_src = self._frame.copy()
            hcol_bgr = (0, 0, 220) if self._side == 'left' else (255, 80, 40)
            for h in self._handles:
                if h is None:
                    continue
                cv.circle(zoom_src, (int(h[0]), int(h[1])), 5, hcol_bgr, -1)
                cv.circle(zoom_src, (int(h[0]), int(h[1])), 6, (255, 255, 255), 1)

            zoomed = get_zoomed_patch(zoom_src, int(hxo), int(hyo),
                                      display_size=self.ZOOM_SZ)
            zrgb   = cv.cvtColor(zoomed, cv.COLOR_BGR2RGB)
            zqimg  = QImage(zrgb.tobytes(), self.ZOOM_SZ, self.ZOOM_SZ,
                            3 * self.ZOOM_SZ, QImage.Format.Format_RGB888)
            zpix   = QPixmap.fromImage(zqimg)

            zx = self.width()  - self.ZOOM_SZ - 6
            zy = 6
            painter.setOpacity(0.90)
            painter.drawPixmap(zx, zy, zpix)
            painter.setOpacity(1.0)
            painter.setPen(QPen(QColor(200, 200, 200), 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(zx, zy, self.ZOOM_SZ, self.ZOOM_SZ)
            painter.setFont(QFont("Segoe UI", 8))
            painter.setPen(QPen(QColor(255, 255, 0), 1))
            painter.drawText(zx + 4, zy + 14, "Zoom (4×)")

        painter.end()

    # ── Souris ────────────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        px, py = event.position().x(), event.position().y()
        hit = self._find_handle(px, py)
        if hit is not None:
            self._drag_idx = hit
            self._hover    = (px, py)
            self.update()
        else:
            ox, oy = self._to_orig(px, py)
            self.point_placed.emit(self._side, ox, oy)

    def mouseMoveEvent(self, event):
        px, py = event.position().x(), event.position().y()
        self._hover = (px, py)
        if self._drag_idx is not None:
            ox, oy = self._to_orig(px, py)
            self._handles[self._drag_idx] = (ox, oy)
        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._drag_idx is not None:
            self._drag_idx = None
            self.handle_released.emit()

    def leaveEvent(self, event):
        self._hover    = None
        self._drag_idx = None
        self.update()


def _cv_to_pixmap(bgr_image: np.ndarray) -> QPixmap:
    rgb   = cv.cvtColor(bgr_image, cv.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg  = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


# ── Onglet Mesure ─────────────────────────────────────────────────────────────

_STEPS = [
    "Click point A on LEFT image",
    "Click point B on LEFT image",
    "Click point A on RIGHT image",
    "Click point B on RIGHT image",
    "All points placed — drag handles to adjust",
]


class MeasureTab(QWidget):
    def __init__(self):
        super().__init__()

        # Vidéo
        self._cap_left:   cv.VideoCapture | None = None
        self._cap_right:  cv.VideoCapture | None = None
        self._frame_left:  np.ndarray | None = None
        self._frame_right: np.ndarray | None = None
        self._total_frames = 0
        self._frame_idx    = 0
        self._playing      = False
        self._left_path    = ""
        self._right_path   = ""

        # Paramètres caméra
        self.mtx1 = self.dist1 = self.mtx2 = self.dist2 = None
        self.R    = self.T = None

        # Mesure — step contrôle le placement de nouveaux points
        self._step = 0

        # ── Layout ────────────────────────────────────────────────────────────
        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.setContentsMargins(12, 12, 12, 12)

        # ── Barre de chargement ───────────────────────────────────────────────
        load_row = QHBoxLayout()
        load_row.setSpacing(6)

        self.btn_load = QPushButton("Load Videos")
        self.btn_load.setToolTip("Charge les vidéos depuis camera_parameters/videos.txt")
        self.btn_load.clicked.connect(self._load_from_saved)
        load_row.addWidget(self.btn_load)

        self.btn_pick_left = QPushButton("Left Video…")
        self.btn_pick_left.clicked.connect(lambda: self._pick_video('left'))
        load_row.addWidget(self.btn_pick_left)

        self.btn_pick_right = QPushButton("Right Video…")
        self.btn_pick_right.clicked.connect(lambda: self._pick_video('right'))
        load_row.addWidget(self.btn_pick_right)

        load_row.addStretch()

        self.btn_disp = QPushButton("Show Disparity Map")
        self.btn_disp.setEnabled(False)
        self.btn_disp.clicked.connect(self._launch_disparity)
        load_row.addWidget(self.btn_disp)

        self.btn_pc = QPushButton("Show 3D Point Cloud")
        self.btn_pc.setEnabled(False)
        self.btn_pc.clicked.connect(self._launch_pointcloud)
        load_row.addWidget(self.btn_pc)

        self.btn_reset = QPushButton("Reset Points")
        self.btn_reset.setEnabled(False)
        self.btn_reset.clicked.connect(self._reset_points)
        load_row.addWidget(self.btn_reset)

        layout.addLayout(load_row)

        # ── Labels état / erreur / hint ───────────────────────────────────────
        self.status_label = QLabel()
        self.status_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.hide()
        layout.addWidget(self.status_label)

        self.hint_label = QLabel()
        self.hint_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hint_label.setStyleSheet("color: #f0c060;")
        self.hint_label.hide()
        layout.addWidget(self.hint_label)

        # ── Images côte à côte ────────────────────────────────────────────────
        img_row = QHBoxLayout()
        img_row.setSpacing(4)
        self.img_left  = MeasureImageWidget('left')
        self.img_right = MeasureImageWidget('right')
        self.img_left.point_placed.connect(self._on_point_placed)
        self.img_right.point_placed.connect(self._on_point_placed)
        self.img_left.handle_released.connect(self._try_compute_distance)
        self.img_right.handle_released.connect(self._try_compute_distance)
        img_row.addWidget(self.img_left,  stretch=1)
        img_row.addWidget(self.img_right, stretch=1)
        layout.addLayout(img_row, stretch=1)

        # ── Slider ────────────────────────────────────────────────────────────
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.sliderMoved.connect(self._on_slider_moved)
        layout.addWidget(self.slider)

        # ── Contrôles lecture ─────────────────────────────────────────────────
        ctrl_row = QHBoxLayout()
        self.btn_play = QPushButton("▶ Play")
        self.btn_play.setFixedWidth(90)
        self.btn_play.setEnabled(False)
        self.btn_play.clicked.connect(self.toggle_play)
        ctrl_row.addWidget(self.btn_play)

        self.lbl_frame = QLabel("Frame: — / —")
        ctrl_row.addWidget(self.lbl_frame)
        ctrl_row.addStretch()

        self.distance_label = QLabel()
        self.distance_label.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        self.distance_label.setStyleSheet("color: #80e080;")
        self.distance_label.hide()
        ctrl_row.addWidget(self.distance_label)
        layout.addLayout(ctrl_row)

        # ── Timer lecture ─────────────────────────────────────────────────────
        self._timer = QTimer()
        self._timer.setInterval(40)   # ~25 fps
        self._timer.timeout.connect(self._next_frame)

        # Auto-load si videos.txt existe
        self._try_auto_load()

    # ── Chargement ────────────────────────────────────────────────────────────

    def _try_auto_load(self):
        if os.path.exists('camera_parameters/videos.txt'):
            try:
                with open('camera_parameters/videos.txt') as f:
                    lines = [l.strip() for l in f.readlines()]
                if len(lines) >= 2 and lines[0] and lines[1]:
                    self._open_videos(lines[0], lines[1])
            except Exception:
                pass

    def _load_from_saved(self):
        if not os.path.exists('camera_parameters/videos.txt'):
            self._show_error("Run calibration first (videos.txt not found)")
            return
        try:
            with open('camera_parameters/videos.txt') as f:
                lines = [l.strip() for l in f.readlines()]
            if len(lines) < 2:
                self._show_error("videos.txt malformed")
                return
            self._open_videos(lines[0], lines[1])
        except Exception as e:
            self._show_error(str(e))

    def _pick_video(self, side: str):
        path, _ = QFileDialog.getOpenFileName(
            self, f"Select {side.capitalize()} Video", "",
            "Videos (*.mp4 *.avi *.mov *.mkv *.MOV);;All Files (*)")
        if not path:
            return
        if side == 'left':
            self._left_path = path
        else:
            self._right_path = path
        if self._left_path and self._right_path:
            self._open_videos(self._left_path, self._right_path)

    def _open_videos(self, left_path: str, right_path: str):
        required = [
            'camera_parameters/mtx1.npy', 'camera_parameters/dist1.npy',
            'camera_parameters/mtx2.npy', 'camera_parameters/dist2.npy',
            'camera_parameters/R.npy',    'camera_parameters/T.npy',
        ]
        if any(not os.path.exists(f) for f in required):
            self._show_error("Run calibration first")
            return

        self.mtx1  = np.load('camera_parameters/mtx1.npy')
        self.dist1 = np.load('camera_parameters/dist1.npy')
        self.mtx2  = np.load('camera_parameters/mtx2.npy')
        self.dist2 = np.load('camera_parameters/dist2.npy')
        self.R     = np.load('camera_parameters/R.npy')
        self.T     = np.load('camera_parameters/T.npy')

        if self._cap_left:
            self._cap_left.release()
        if self._cap_right:
            self._cap_right.release()

        self._cap_left  = cv.VideoCapture(left_path)
        self._cap_right = cv.VideoCapture(right_path)

        if not self._cap_left.isOpened() or not self._cap_right.isOpened():
            self._show_error(f"Cannot open videos:\n  {left_path}\n  {right_path}")
            return

        n_left  = int(self._cap_left.get(cv.CAP_PROP_FRAME_COUNT))
        n_right = int(self._cap_right.get(cv.CAP_PROP_FRAME_COUNT))
        self._total_frames = min(n_left, n_right)

        self.slider.setRange(0, max(0, self._total_frames - 1))
        self._frame_idx = 0
        self._seek(0)

        self.btn_play.setEnabled(True)
        self.btn_reset.setEnabled(True)
        self.btn_pc.setEnabled(True)
        self.btn_disp.setEnabled(True)
        self._reset_points()
        self.hint_label.show()
        self.status_label.hide()

    # ── Lecture vidéo ─────────────────────────────────────────────────────────

    def _seek(self, idx: int):
        idx = max(0, min(idx, self._total_frames - 1))
        self._frame_idx = idx
        self._cap_left.set(cv.CAP_PROP_POS_FRAMES, idx)
        self._cap_right.set(cv.CAP_PROP_POS_FRAMES, idx)
        ret1, f1 = self._cap_left.read()
        ret2, f2 = self._cap_right.read()
        if ret1 and ret2:
            self._frame_left  = f1
            self._frame_right = f2
            self._refresh_display()
        self.slider.setValue(idx)
        self.lbl_frame.setText(f"Frame: {idx} / {self._total_frames - 1}")

    def _next_frame(self):
        if self._cap_left is None:
            return
        ret1, f1 = self._cap_left.read()
        ret2, f2 = self._cap_right.read()
        if not ret1 or not ret2:
            self._timer.stop()
            self._playing = False
            self.btn_play.setText("▶ Play")
            return
        self._frame_left  = f1
        self._frame_right = f2
        self._frame_idx  += 1
        self.slider.setValue(self._frame_idx)
        self.lbl_frame.setText(f"Frame: {self._frame_idx} / {self._total_frames - 1}")
        self._refresh_display()

    def _on_slider_moved(self, val: int):
        if self._playing:
            self._timer.stop()
            self._playing = False
            self.btn_play.setText("▶ Play")
        if self._cap_left:
            self._seek(val)

    def toggle_play(self):
        if self._cap_left is None:
            return
        if self._playing:
            self._timer.stop()
            self._playing = False
            self.btn_play.setText("▶ Play")
        else:
            self._timer.start()
            self._playing = True
            self.btn_play.setText("⏸ Pause")

    # ── Affichage ─────────────────────────────────────────────────────────────

    def _refresh_display(self):
        if self._frame_left is not None:
            self.img_left.set_frame(self._frame_left)
        if self._frame_right is not None:
            self.img_right.set_frame(self._frame_right)

    # ── Mesure par clics / handles ────────────────────────────────────────────

    def _on_point_placed(self, side: str, x: float, y: float):
        """Reçu depuis MeasureImageWidget quand l'user clique hors d'un handle."""
        if self._frame_left is None:
            return
        if side == 'left' and self._step in (0, 1):
            if self._playing:
                return
            self.img_left.set_handle(self._step, x, y)
            self._step += 1
            self._update_hint()
        elif side == 'right' and self._step in (2, 3):
            if self._playing:
                return
            self.img_right.set_handle(self._step - 2, x, y)
            self._step += 1
            self._update_hint()
            if self._step == 4:
                self._try_compute_distance()

    def _try_compute_distance(self):
        """Recompute si les 4 handles sont posés (appelé aussi après drag)."""
        left_h  = self.img_left.get_handles()
        right_h = self.img_right.get_handles()
        if None in left_h or None in right_h:
            return
        if self.mtx1 is None:
            return
        RT1  = np.concatenate([np.eye(3), [[0], [0], [0]]], axis=-1)
        P1   = self.mtx1 @ RT1
        RT2  = np.concatenate([self.R, self.T], axis=-1)
        P2   = self.mtx2 @ RT2
        p3dA = DLT(P1, P2, list(left_h[0]), list(right_h[0]))
        p3dB = DLT(P1, P2, list(left_h[1]), list(right_h[1]))
        dist = np.linalg.norm(p3dB - p3dA)
        self.distance_label.setText(f"Distance A→B : {dist:.4f} mm")
        self.distance_label.show()

    def _update_hint(self):
        self.hint_label.setText(_STEPS[min(self._step, 4)])

    def _reset_points(self):
        self._step = 0
        self.img_left.clear_handles()
        self.img_right.clear_handles()
        self.hint_label.setText(_STEPS[0])
        self.distance_label.hide()
        self.status_label.hide()

    # ── Nuage de points 3D ────────────────────────────────────────────────────

    # ── Disparity Map ─────────────────────────────────────────────────────────

    def _launch_disparity(self):
        if self._frame_left is None:
            return
        if self._playing:
            self._show_error("Pause the video first")
            return

        self._show_status("Computing disparity map…")
        self.btn_disp.setEnabled(False)

        frame1 = self._frame_left.copy()
        frame2 = self._frame_right.copy()
        mtx1, dist1 = self.mtx1, self.dist1
        mtx2, dist2 = self.mtx2, self.dist2
        R, T = self.R, self.T

        self._disp_signals = _PCSignals()
        self._disp_signals.status.connect(self._show_status)
        self._disp_signals.error.connect(self._show_error)
        self._disp_signals.finished.connect(
            lambda: self.btn_disp.setEnabled(True))
        self._disp_signals.image_ready.connect(self._open_disparity_window)
        signals = self._disp_signals

        def compute_and_show():
            try:
                h, w = frame1.shape[:2]
                size = (w, h)

                # ── 1. Rectification ─────────────────────────────────────────
                R1, R2, P1r, P2r, Q, _, _ = cv.stereoRectify(
                    mtx1, dist1, mtx2, dist2, size, R, T, alpha=0)
                map1x, map1y = cv.initUndistortRectifyMap(
                    mtx1, dist1, R1, P1r, size, cv.CV_32FC1)
                map2x, map2y = cv.initUndistortRectifyMap(
                    mtx2, dist2, R2, P2r, size, cv.CV_32FC1)
                rect1 = cv.remap(frame1, map1x, map1y, cv.INTER_LINEAR)
                rect2 = cv.remap(frame2, map2x, map2y, cv.INTER_LINEAR)

                # ── 2. Disparity ──────────────────────────────────────────────
                stereo = cv.StereoSGBM_create(
                    minDisparity=0, numDisparities=128, blockSize=5,
                    P1=8 * 3 * 25, P2=32 * 3 * 25,
                    disp12MaxDiff=1, uniquenessRatio=10,
                    speckleWindowSize=100, speckleRange=2,
                    mode=cv.STEREO_SGBM_MODE_SGBM_3WAY)
                disp_raw = stereo.compute(
                    cv.cvtColor(rect1, cv.COLOR_BGR2GRAY),
                    cv.cvtColor(rect2, cv.COLOR_BGR2GRAY),
                ).astype(np.float32) / 16.0

                # ── 3. Stats ─────────────────────────────────────────────────
                valid_mask = disp_raw > 0
                pct_valid  = 100.0 * valid_mask.sum() / disp_raw.size
                d_min  = float(disp_raw[valid_mask].min())  if valid_mask.any() else 0
                d_max  = float(disp_raw[valid_mask].max())  if valid_mask.any() else 0
                d_mean = float(disp_raw[valid_mask].mean()) if valid_mask.any() else 0
                signals.status.emit(
                    f"Disparity — mean: {d_mean:.1f}px | "
                    f"min: {d_min:.1f} | max: {d_max:.1f} | "
                    f"valid pixels: {pct_valid:.1f}%")

                # ── 4. Colormap TURBO (noir = invalide) ───────────────────────
                disp_norm = np.zeros_like(disp_raw, dtype=np.uint8)
                if valid_mask.any():
                    cv.normalize(disp_raw, disp_norm, 0, 255,
                                 cv.NORM_MINMAX, mask=valid_mask.astype(np.uint8))
                disp_color = cv.applyColorMap(disp_norm, cv.COLORMAP_TURBO)
                disp_color[~valid_mask] = (0, 0, 0)   # pixels invalides en noir

                # ── 5. Lignes épipolaires horizontales ───────────────────────
                def draw_epilines(img):
                    out = img.copy()
                    for y in range(0, h, 30):
                        cv.line(out, (0, y), (w, y), (0, 220, 0), 1,
                                cv.LINE_AA)
                    return out

                vis_l = draw_epilines(rect1)
                vis_r = draw_epilines(rect2)

                # ── 6. Composition côte à côte ───────────────────────────────
                label_h = 28
                def add_label(img, text):
                    out = np.zeros((img.shape[0] + label_h, img.shape[1], 3),
                                   dtype=np.uint8)
                    out[label_h:] = img
                    cv.putText(out, text, (8, 20),
                               cv.FONT_HERSHEY_SIMPLEX, 0.65,
                               (200, 200, 200), 1, cv.LINE_AA)
                    return out

                panel_l = add_label(vis_l,    "LEFT rectified + epipolar lines")
                panel_c = add_label(disp_color, "DISPARITY MAP (turbo | black=invalid)")
                panel_r = add_label(vis_r,    "RIGHT rectified + epipolar lines")

                # Redimensionner à hauteur commune si nécessaire
                target_h = panel_l.shape[0]
                def resize_h(img, th):
                    ratio = th / img.shape[0]
                    return cv.resize(img,
                                     (int(img.shape[1] * ratio), th),
                                     interpolation=cv.INTER_LINEAR)

                composite = np.hstack([
                    resize_h(panel_l, target_h),
                    np.full((target_h, 4, 3), 40, dtype=np.uint8),  # séparateur
                    resize_h(panel_c, target_h),
                    np.full((target_h, 4, 3), 40, dtype=np.uint8),
                    resize_h(panel_r, target_h),
                ])

                signals.image_ready.emit(composite)

            except Exception as exc:
                import traceback
                signals.error.emit(f"[DISP ERROR] {exc}\n{traceback.format_exc()}")
            finally:
                signals.finished.emit()

        threading.Thread(target=compute_and_show, daemon=True).start()

    def _open_disparity_window(self, bgr_image: np.ndarray):
        """Ouvre la fenêtre de disparité dans le thread principal (Qt)."""
        self._disp_win = DisparityWindow(bgr_image)
        self._disp_win.show()

    # ── Nuage de points 3D ────────────────────────────────────────────────────

    def _launch_pointcloud(self):
        if not _O3D_AVAILABLE:
            self._show_error("pip install open3d")
            return
        if self._frame_left is None:
            return
        if self._playing:
            self._show_error("Pause the video first")
            return

        self._show_status("Computing point cloud…")
        self.btn_pc.setEnabled(False)

        frame1 = self._frame_left.copy()
        frame2 = self._frame_right.copy()
        mtx1, dist1 = self.mtx1, self.dist1
        mtx2, dist2 = self.mtx2, self.dist2
        R, T = self.R, self.T

        self._pc_signals = _PCSignals()
        self._pc_signals.status.connect(self._show_status)
        self._pc_signals.error.connect(self._show_error)
        self._pc_signals.finished.connect(lambda: self.btn_pc.setEnabled(True))
        signals = self._pc_signals

        def compute_and_show():
            try:
                R1, R2, P1r, P2r, Q, _, _ = cv.stereoRectify(
                    mtx1, dist1, mtx2, dist2,
                    frame1.shape[:2][::-1], R, T, alpha=0)
                map1x, map1y = cv.initUndistortRectifyMap(
                    mtx1, dist1, R1, P1r, frame1.shape[:2][::-1], cv.CV_32FC1)
                map2x, map2y = cv.initUndistortRectifyMap(
                    mtx2, dist2, R2, P2r, frame2.shape[:2][::-1], cv.CV_32FC1)
                rect1 = cv.remap(frame1, map1x, map1y, cv.INTER_LINEAR)
                rect2 = cv.remap(frame2, map2x, map2y, cv.INTER_LINEAR)

                stereo = cv.StereoSGBM_create(
                    minDisparity=0, numDisparities=128, blockSize=5,
                    P1=8 * 3 * 25, P2=32 * 3 * 25,
                    disp12MaxDiff=1, uniquenessRatio=10,
                    speckleWindowSize=100, speckleRange=2,
                    mode=cv.STEREO_SGBM_MODE_SGBM_3WAY)
                disparity = stereo.compute(
                    cv.cvtColor(rect1, cv.COLOR_BGR2GRAY),
                    cv.cvtColor(rect2, cv.COLOR_BGR2GRAY),
                ).astype(np.float32) / 16.0

                points_3d = cv.reprojectImageTo3D(disparity, Q)
                mask = ((disparity > disparity.min()) &
                        (points_3d[:, :, 2] > 0) &
                        (points_3d[:, :, 2] < 100000))
                points = points_3d[mask]
                colors = rect1[mask][:, ::-1].astype(np.float64) / 255.0

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
                pcd.colors = o3d.utility.Vector3dVector(colors)
                pcd = pcd.voxel_down_sample(voxel_size=2.0)

                signals.status.emit("Point cloud window opened")
                o3d.visualization.draw_geometries(
                    [pcd], window_name="3D Point Cloud",
                    width=1024, height=768)

            except Exception as exc:
                signals.error.emit(f"[ERROR]  {exc}")
            finally:
                signals.finished.emit()

        threading.Thread(target=compute_and_show, daemon=True).start()

    # ── Statut / erreur ───────────────────────────────────────────────────────

    def _show_status(self, msg: str):
        self.status_label.setText(msg)
        self.status_label.setStyleSheet("color: #5ab5f5; font-weight: bold;")
        self.status_label.show()

    def _show_error(self, msg: str):
        self.status_label.setText(msg)
        self.status_label.setStyleSheet("color: #e05555; font-weight: bold;")
        self.status_label.show()


# ── Fenêtre principale ────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AquaMeasure")
        self.resize(1100, 780)

        # ── Widget central ────────────────────────────────────────────────────
        central = QWidget()
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ── Bandeau titre ─────────────────────────────────────────────────────
        header = QWidget()
        header.setStyleSheet("background:#0d3b6e;")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(16, 10, 16, 10)
        header_layout.setSpacing(2)

        title_lbl = QLabel("AquaMeasure")
        title_lbl.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        title_lbl.setStyleSheet("color:#ffffff;")
        title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(title_lbl)

        sub_lbl = QLabel("Stereo vision fish measurement tool")
        sub_lbl.setFont(QFont("Segoe UI", 11))
        sub_lbl.setStyleSheet("color:#a0c4e8;")
        sub_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(sub_lbl)

        root_layout.addWidget(header)

        # ── Onglets ───────────────────────────────────────────────────────────
        self._tabs = QTabWidget()
        self._calib_tab   = CalibrationTab()
        self._measure_tab = MeasureTab()
        self._tabs.addTab(self._calib_tab,   "Calibration")
        self._tabs.addTab(self._measure_tab, "Measure")
        root_layout.addWidget(self._tabs, stretch=1)

        self.setCentralWidget(central)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self._measure_tab.toggle_play()
        else:
            super().keyPressEvent(event)


# ── Point d'entrée ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
